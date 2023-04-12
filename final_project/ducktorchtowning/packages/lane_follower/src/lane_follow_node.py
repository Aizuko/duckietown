#!/usr/bin/env python3
import rospy
import cv2
import cv2 as cv
import time
import json

from duckietown.dtros import DTROS, NodeType
from dataclasses import dataclass
from enum import IntEnum, auto, unique
from sensor_msgs.msg import CameraInfo, CompressedImage, Range
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import numpy as np
from nav_msgs.msg import Odometry
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
from geometry_msgs.msg import Transform, Vector3, TransformStamped
from std_msgs.msg import Float64
from tf2_ros import Buffer, TransformListener
from tf import transformations as tr

ROAD_MASK = [(10, 60, 165), (40, 255, 255)]
DUCKIES_PLUS_LINE = [(0, 70, 120), (40, 255, 255)]
DUCKIES_ONLY = [(0, 55, 145), (20, 255, 255)]
REDLINE_MASK = [(0, 100, 120), (10, 255, 255)]
BLUELINE_MASK = [(40, 100, 80), (130, 255, 255)]
PARKING_LANE_MASK = [(22, 70, 120), (40, 255, 255)]
# Crop off top 270 for lines
# Crop off top 340 for real closeness

with open("/params.json") as f:
    params = json.load(f)["default"]


@unique
class DS(IntEnum):
    """
    Statemachine, segmented by project stage
    """

    Stage1Loops = 10
    Stage1Loops_LaneFollowing = 11
    Stage1Loops_ForceForward = 12
    Stage1Loops_ForceRight = 13
    Stage1Loops_ForceLeft = 14
    Stage1Loops_ThinkDuck = 19

    Stage2Ducks = 20
    Stage2Ducks_LaneFollowing = 21
    Stage2Ducks_WaitForCrossing = 22
    Stage2Ducks_WaitToHelp = 23
    Stage2Ducks_LondonStyle = 24
    Stage2Ducks_ThinkDuck = 29

    Stage3Parking = 30
    Stage3Parking_Forward = 31
    Stage3Parking_Turn = 32
    Stage3Parking_Reverse = 33
    Stage3Parking_ThinkDuck = 39

    ShuttingDown = 90


class TagType(IntEnum):
    """IntEnum mirror of the classification @ apriltag tag.py"""

    NotImportant = 0
    RightStop = 1
    LeftStop = 2
    ForwardStop = 3
    CrossingStop = 4
    ParkingLotEnteringStop = 5


class SeenAP:
    """Tuple for an apriltag detection"""

    def __init__(self, tag: TagType, distance: float):
        self.tag = tag
        self.distance = distance
        self.time = time.time()

    def is_within_time(self) -> bool:
        return time.time() - self.time < params["ap_stale_timeout"]

    def is_within_distance(self) -> bool:
        return self.distance < params["ap_considered_distance"]

    def is_within_criteria(self) -> bool:
        return self.is_within_time() and self.is_within_distance()


class LaneFollowNode(DTROS):
    def __init__(self, node_name):
        super(LaneFollowNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC
        )
        self.node_name = node_name
        self.bridge = CvBridge()
        self.veh = rospy.get_param("~veh")

        self.params = params

        # ╔─────────────────────────────────────────────────────────────────────╗
        # │ Sτατε cδητrδls                                                      |
        # ╚─────────────────────────────────────────────────────────────────────╝
        self.state = DS(self.params["starting_state"])
        self.state_start_time = time.time()
        self.image = None
        self.is_new_image = False

        self.is_parked = False
        self.duck_free_time = 0.0
        self.last_seen_duck = None
        self.seen_ap = [None, None]

        # ╔─────────────────────────────────────────────────────────────────────╗
        # │ Lαηε fδllδωiηg PID sεττiηgs                                         |
        # ╚─────────────────────────────────────────────────────────────────────╝
        self.lane_offset = self.params["lane_offset"]

        self.velocity = self.params["lane_follow_velocity"]
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        self.bottom_error = None
        self.right_error = None
        self.left_error = None

        self.right_last_error = 0
        self.left_last_error = 0
        self.bottom_last_error = 0
        self.red_far_sightings = [None, None]
        self.red_close_sightings = [None, None]

        self.right_last_time = rospy.get_time()
        self.left_last_time = rospy.get_time()
        self.bottom_last_time = rospy.get_time()

        self.P = self.params["Px_coef"]
        self.D = self.params["Dx_coef"]
        self.odom = None

        # For stage 2 robot detection
        self.tof_dist_list = [0.0, 0.0, 0.0]
        self.is_ready_to_help = False
        self.is_helping = False
        self.has_helped = False
        self.started_waiting_for_help_time = None
        self.started_english_time = None

        # ╔─────────────────────────────────────────────────────────────────────╗
        # │ Pαrkiηg αττribμτεs                                                  |
        # ╚─────────────────────────────────────────────────────────────────────╝
        self.parking_last_error = 0
        self.parking_last_time = rospy.get_time()

        parking_lot = self.params["parking_lot"]
        parking_stall_number = self.params["parking_stall_number"]

        self.parking_stall = parking_lot[parking_stall_number - 1]
        opposite_stall_number = self.parking_stall["opposite_stall_number"]
        self.opposite_stall = parking_lot[opposite_stall_number - 1]
        self.tof_distance = np.inf

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        # ╔─────────────────────────────────────────────────────────────────────╗
        # │ Pμblishεrs & Sμbscribεrs                                            |
        # ╚─────────────────────────────────────────────────────────────────────╝
        self.pub = rospy.Publisher(
            f"/{self.veh}/output/image/mask/compressed",
            CompressedImage,
            queue_size=1,
        )
        self.pub_red = rospy.Publisher(
            f"/{self.veh}/output/image/red/compressed",
            CompressedImage,
            queue_size=1,
        )
        self.pub_parking_lane = rospy.Publisher(
            f"/{self.veh}/output/image/parking_lane/compressed",
            CompressedImage,
            queue_size=1,
        )
        self.vel_pub = rospy.Publisher(
            f"/{self.veh}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1
        )
        self.ap_sub = rospy.Subscriber(
            f"/{self.veh}/ap_node/ap_detection",
            Vector3,
            self.ap_callback,
            queue_size=1,
        )
        self.sub = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.lane_callback,
            queue_size=1,
            buff_size="20MB",
        )
        self.odom_sub = rospy.Subscriber(
            f"/{self.veh}/deadreckoning_node/odom",
            Odometry,
            self.odom_callback,
            queue_size=1,
        )
        self.tof_sub = rospy.Subscriber(
            f"/{self.veh}/front_center_tof_driver_node/range",
            Range,
            self.tof_callback,
            queue_size=1,
        )
        self.robot_ahead_transform_sub = rospy.Subscriber(
            f"/{self.veh}/duckiebot_distance_node/transform",
            TransformStamped,
            self.robot_ahead_transform_callback,
            queue_size=1,
        )

        self.timer = rospy.Timer(rospy.Duration(1), self.debug_callback)

        self.image = None

        # Shutdown hook
        rospy.on_shutdown(self.hook)

    def debug_callback(self, _):
        if not self.params["is_debug"]:
            return
        self.parking_yellow_lane_error()
        rospy.loginfo(f"April tags: {len(self.seen_ap)}")

        if self.seen_ap[-1] is not None:
            rospy.loginfo(f"latest tag tag        {self.seen_ap[-1].tag.name}")
            rospy.loginfo(
                f"Delta from latest tag {time.time() - self.seen_ap[-1].time}"
            )
            rospy.loginfo(f"Dist from latest tag {self.seen_ap[-1].distance}")

        rospy.loginfo(f"==== State: {self.state.name} ====")
        rospy.loginfo(f"Red far: {len(self.red_far_sightings)}")
        rospy.loginfo(f"Red close: {len(self.red_close_sightings)}")

    def state_decision(self):
        if self.is_parked:
            self.state = DS.ShuttingDown
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("End of the road")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            rospy.signal_shutdown("Rode to the end of the road")

        if 30 <= self.state < 40:
            return  # Let parking figure out its own state

        if self.is_ready_to_help and not self.is_helping:
            if (
                time.time() - self.started_waiting_for_help_time
                > self.params["wait_for_help_time"]
            ):
                self.state = DS.Stage2Ducks_WaitToHelp
                self.is_helping = True
                self.started_english_time = time.time()
                self.offset *= -1
            else:
                self.state = DS.Stage2Ducks_WaitToHelp
            return
        elif self.is_helping and not self.has_helped:
            if (
                time.time() - self.started_english_time
                > self.params["english_time"]
            ):
                self.state = DS.Stage2Ducks_LaneFollowing
                self.has_helped = True
                self.offset *= -1
            else:
                self.state = DS.Stage2Ducks_LondonStyle
            return

        # Make decision by ap node
        for i in range(-1, -(10**6), -1):
            try:
                ap = self.seen_ap[i]

                if ap is None:
                    break

                rospy.logwarn(f"==== Ap type: {ap.tag.name}")

                if not ap.is_within_time():
                    rospy.logwarn(
                        f"Broken on time delta: {time.time() - ap.time}"
                    )
                    break
                elif not ap.is_within_distance():
                    rospy.logwarn(f"Broken on distance delta: {ap.distance}")
                    break
                elif ap.tag == TagType.ParkingLotEnteringStop:
                    self.state = DS.Stage3Parking_ThinkDuck
                elif self.state == DS.Stage1Loops_LaneFollowing:
                    if ap.tag == TagType.RightStop:
                        self.state = DS.Stage1Loops_ForceRight
                    elif ap.tag == TagType.LeftStop:
                        self.state = DS.Stage1Loops_ForceLeft
                    elif ap.tag == TagType.ForwardStop:
                        self.state = DS.Stage1Loops_ForceForward
                    elif ap.tag == TagType.CrossingStop:
                        self.state = DS.Stage2Ducks_WaitForCrossing
                elif 20 <= self.state < 30 and ap.tag == TagType.CrossingStop:
                    self.state = DS.Stage2Ducks_WaitForCrossing
                else:
                    rospy.logwarn("NO UPDATE")
            except IndexError:
                break

    def ap_callback(self, msg):
        # Don't do any ap callbacks in the parking state
        if DS.Stage3Parking <= self.state < DS.Stage3Parking + 10:
            return

        self.seen_ap.append(SeenAP(TagType(int(msg.y)), msg.x))

    def tof_callback(self, msg):
        self.tof_dist_list.append(msg.range)
        self.tof_distance = min(msg.range, self.params["max_tof_distance"])

    def lane_callback(self, msg):
        self.image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        self.is_new_image = True

    def robot_ahead_transform_callback(self, msg: TransformStamped):
        if self.is_ready_to_help or self.has_helped:
            return

        transform = msg.transform
        T = tr.compose_matrix(
            translate=(
                [
                    transform.translation.x,
                    transform.translation.y,
                    transform.translation.z,
                ]
            ),
            angles=tr.euler_from_quaternion(
                [
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.rotation.w,
                ]
            ),
        )

        latest_translate = T[:3, 3]

        tof_dist_transformed = (
            self.params["tof_a"] * self.tof_dist_list[-1] + self.params["tof_b"]
        )

        robot_dist = p.min(
            np.linalg.norm(latest_translate), tof_dist_transformed
        )

        if robot_dist < self.params["min_robot_dist"]:
            self.is_ready_to_help = True

    def odom_callback(self, msg):
        self.odom = msg

    def evaluate_errors(self):
        """What was previously in lane_callback"""
        if self.image is None or 30 <= self.state < 40:
            return

        if not self.is_new_image:  # Don't re-eval same image
            return

        self.is_new_image = False
        image = self.image

        if self.state == DS.Stage2Ducks_WaitForCrossing:
            if not self.is_good2go(image):
                return
            self.state = DS.Stage2Ducks_LaneFollowing

        right_image = image[:, 400:, :]
        left_image = image[:, :-400, :]
        bottom_image = image[300:-1, :, :]
        lines_far_image = image[340:, :, :]
        lines_close_image = image[270:, :, :]

        crop_width = bottom_image.shape[1]

        # Right
        hsv = cv2.cvtColor(right_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        right_image = cv2.bitwise_and(right_image, right_image, mask=mask)
        right_conts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Left
        hsv = cv2.cvtColor(left_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        left_image = cv2.bitwise_and(left_image, left_image, mask=mask)
        left_conts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Bottom
        hsv = cv2.cvtColor(bottom_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        bottom_image = cv2.bitwise_and(bottom_image, bottom_image, mask=mask)
        bottom_conts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Line far red
        hsv = cv2.cvtColor(lines_far_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, REDLINE_MASK[0], REDLINE_MASK[1])
        redline_far_image = cv2.bitwise_and(
            lines_far_image, lines_far_image, mask=mask
        )
        red_far_conts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Line close red
        hsv = cv2.cvtColor(lines_close_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, REDLINE_MASK[0], REDLINE_MASK[1])
        redline_close_image = cv2.bitwise_and(
            lines_close_image, lines_close_image, mask=mask
        )
        red_close_conts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        right_areas = np.array([cv2.contourArea(a) for a in right_conts])
        left_areas = np.array([cv2.contourArea(a) for a in left_conts])
        bottom_areas = np.array([cv2.contourArea(a) for a in bottom_conts])
        red_far_areas = np.array([cv2.contourArea(a) for a in red_far_conts])
        red_close_areas = np.array(
            [cv2.contourArea(a) for a in red_close_conts]
        )

        # Right error
        if len(right_areas) == 0 or np.max(right_areas) < 20:
            self.right_error = None
        else:
            max_idx = np.argmax(right_areas)

            M = cv2.moments(right_conts[max_idx])
            try:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                self.right_error = cx - int(crop_width / 2) + self.lane_offset
            except:
                pass

        # Left error
        if len(left_areas) == 0 or np.max(left_areas) < 20:
            self.left_error = None
        else:
            max_idx = np.argmax(left_areas)

            M = cv2.moments(left_conts[max_idx])
            try:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                self.left_error = cx - int(crop_width / 2) + self.lane_offset
            except:
                pass

        # Bottom error
        if len(bottom_areas) == 0 or np.max(bottom_areas) < 20:
            self.bottom_error = None
        else:
            max_idx = np.argmax(bottom_areas)

            M = cv2.moments(bottom_conts[max_idx])

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            self.bottom_error = cx - int(crop_width / 2) + self.lane_offset

        # Red far
        if (
            len(red_far_areas) > 0
            and np.max(red_far_areas) > self.params["red_far_thresh"]
        ):
            self.red_far_sightings.append(time.time())

        # Red close
        if (
            len(red_close_areas) > 0
            and np.max(red_close_areas) > self.params["red_close_thresh"]
        ):
            self.red_close_sightings.append(time.time())

    def is_good2go(self, image):
        if (
            self.last_seen_duck is not None
            and (time.time() - self.last_seen_duck)
            > self.params["crossing_timeout"]
        ):
            self.last_seen_duck = None

        image = image[200:, :, :]  # Tends to be better to grab only half

        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, DUCKIES_ONLY[0], DUCKIES_ONLY[1])

        image = cv.bitwise_and(image, image, mask=mask)
        image = cv.cvtColor(image[200:, :, :], cv.COLOR_BGR2GRAY)
        image[image != 0] = 1

        is_occupied = np.sum(image) > self.params["duckie_crossing_thresh"]

        if is_occupied:
            self.last_seen_duck = time.time()
            return False
        elif self.last_seen_duck is None:
            return True
        elif (time.time() - self.last_seen_duck) > self.params[
            "crossing_wait_time"
        ]:
            return True
        else:
            return False

    def drive(self):
        delta_t = time.time() - self.state_start_time
        # rospy.loginfo_throttle(
        #     2,
        #     f"Errors: {self.left_error}, {self.right_error}, {self.bottom_error}",
        # )
        rospy.loginfo_throttle(1, f"State: {self.state.name}")

        # ==== Set target ====
        if self.state == DS.Stage1Loops_LaneFollowing:
            self.twist.v = self.params["lane_follow_velocity"]
            self.twist.omega = self.get_lanefollowing_omega()
        elif self.state == DS.Stage1Loops_ForceForward:
            self.twist.v = self.params["forward_turn_velocity"]
            self.twist.omega = self.params["forward_turn_omega"]
        elif self.state == DS.Stage1Loops_ForceRight:
            self.twist.v = self.params["right_turn_velocity"]
            self.twist.omega = self.params["right_turn_omega"]
        elif self.state == DS.Stage1Loops_ForceLeft:
            self.twist.v = self.params["left_turn_velocity"]
            self.twist.omega = self.params["left_turn_omega"]
        elif self.state == DS.Stage1Loops_ThinkDuck:
            self.twist.v = self.twist.omega = 0
        elif self.state == DS.Stage2Ducks_LaneFollowing:
            self.twist.v = self.params["lane_follow_velocity"]
            self.twist.omega = self.get_lanefollowing_omega()
        elif self.state == DS.Stage2Ducks_WaitForCrossing:
            self.twist.v = self.twist.omega = 0
        elif self.state == DS.Stage2Ducks_WaitToHelp:
            self.twist.v = self.twist.omega = 0
        elif self.state == DS.Stage2Ducks_LondonStyle:
            self.twist.v = self.params["lane_follow_velocity"]
            self.twist.omega = self.get_lanefollowing_omega()
        elif self.state == DS.Stage3Parking_ThinkDuck:
            self.twist.v = self.twist.omega = 0
            self.parking_stop_state()
        elif self.state == DS.Stage3Parking_Forward:
            self.parking_forward_state()
        elif self.state == DS.Stage3Parking_Turn:
            self.parking_turn_state()
        elif self.state == DS.Stage3Parking_Reverse:
            self.parking_reverse_state()
        elif self.state == DS.ShuttingDown:
            self.twist.v = self.twist.omega = 0
        else:
            print(f"Found unknown state: {self.state.name}")
            rospy.signal_shutdown("Reached unknown state")

        self.vel_pub.publish(self.twist)

    def get_lanefollowing_omega(self):
        """Justin's pid for lane following"""
        if self.bottom_error is None:
            return 0

        # P Term
        P = -self.bottom_error * self.P

        # D Term
        d_error = (self.bottom_error - self.bottom_last_error) / (
            rospy.get_time() - self.bottom_last_time
        )
        self.bottom_last_error = self.bottom_error
        self.bottom_last_time = rospy.get_time()
        D = d_error * self.D

        return P + D

    def hook(self):
        print("SHUTTING DOWN")
        self.twist.v = self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)

    def parking_stop_state(self):
        self.state_start_time = time.time()
        rate = rospy.Rate(self.params["parking_rate"])
        while time.time() - self.state_start_time < self.params["parking_stop_time"] or self.odom is None:
            self.twist.v = 0
            self.twist.omega = 0
            self.vel_pub.publish(self.twist)
            rate.sleep()
        self.state = DS.Stage3Parking_Forward

    def parking_pid(self, error, P_, D_):
        P = error * P_
        d_error = error - self.parking_last_error / (
            rospy.get_time() - self.parking_last_time
        )
        self.parking_last_error = error
        self.parking_last_time = rospy.get_time()
        D = d_error * D_
        v = P[0] + D[0]
        v = np.clip(
            v, -self.params["parking_max_v"], self.params["parking_max_v"]
        )
        omega = P[1] + D[1]
        omega = np.clip(
            omega,
            -self.params["parking_max_omega"],
            self.params["parking_max_omega"]
        )
        self.twist.v = v
        self.twist.omega = omega

    def filter_parking_contours(self, contour):
        # ratio of contour y length to x length
        x, y, w, h = cv.boundingRect(contour)
        if h == 0:
            return False
        aspect_ratio = w /h
        return aspect_ratio > self.params["parking_lane_aspect_ratio"]

    def parking_yellow_lane_error(self):
        if self.image is None:
            return None
        image = self.image[self.params["parking_lane_crop_top"]:, :]
        image = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, DUCKIES_PLUS_LINE[0], DUCKIES_PLUS_LINE[1])
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        BLACK = (0, 0, 0)
        if self.params["parking_lane_debug"]:
            image_copy = image.copy()
            cv2.drawContours(image_copy, contours, -1, BLACK, 3)
            for contour in contours:
                area = cv2.contourArea(contour)
                cv2.putText(
                    image_copy,
                    f"{int(area)}",
                    tuple(contour[0][0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    BLACK,
                    2,
                )

        contours = list(filter(lambda x: cv2.contourArea(x) > self.params["parking_lane_min_area"], contours))
        if len(contours) < 3:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = contours[:3]
        contours = sorted(contours, key=lambda x: x[0][0][0])
        if self.parking_stall["depth"] == "near":
            contours = contours[-2:]
        elif self.parking_stall["depth"] == "far":
            contours = contours[:2]

        # get parallel lines
        lines = []

        colors = [(255, 0, 0), (0, 0, 255)]

        for contour, color in zip(contours, colors):
            # https://stackoverflow.com/questions/64396183/opencv-find-a-middle-line-of-a-contour-python
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return None
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            theta = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
            endx = np.cos(theta) + center[0]
            endy = np.sin(theta) + center[1]
            line = np.cross([center[0], center[1], 1], [endx, endy, 1])
            lines.append(line)
            if self.params["parking_lane_debug"]:
                cv2.drawContours(image_copy, [contour], -1, color, 3)
                area = cv2.contourArea(contour)
                cv2.putText(
                    image_copy,
                    f"{int(area)}",
                    (center),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2
                )
                cv2.line(image_copy,
                         center,
                         (int(endx), int(endy)), BLACK, 3)
        lines = np.array(lines)

        # get point at infinity at intersection
        intersection = np.cross(lines[0], lines[1])
        intersection /= intersection[2]
        intersection = intersection.astype(int)

        # error is angle of intersection with vertical line
        error = np.arctan2(
            intersection[0] - image.shape[1] // 2,
            abs(image.shape[0] - intersection[1]),
        )

        if self.params["parking_lane_debug"]:
            cv2.circle(image_copy, (intersection[0], intersection[1]), 10, (255, 0, 255), -1)
            cv2.line(image_copy, (intersection[0], 0), (intersection[0], image.shape[0]), (0, 255, 0), 3)
            cv2.line(image_copy, (image.shape[1] // 2, 0), (image.shape[1] // 2, image.shape[0]), (255, 255, 0), 3)
            cv2.putText(image_copy, f"{error:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 2, cv2.LINE_AA)
            message = self.bridge.cv2_to_compressed_imgmsg(image_copy, dst_format="jpeg")
            self.pub_parking_lane.publish(message)

        return error

    def parking_forward_state(self):
        rate = rospy.Rate(self.params["parking_rate"])
        P_ = np.array(
            [self.params["parking_P_x"], 0]
        )
        D_ = np.array(
            [self.params["parking_D_x"], 0]
        )
        depth = self.parking_stall["depth"]
        while True:
            distance_error = (
                self.tof_distance - self.params["parking_forward_distance"][depth]
            )
            error = np.array([distance_error, 0])
            if abs(distance_error) < self.params["parking_forward_epsilon"]:
                break
            self.parking_pid(error, P_, D_)
            self.vel_pub.publish(self.twist)
            rate.sleep()

        self.state = DS.Stage3Parking_Turn

    def parking_turn_state(self):
        rate = rospy.Rate(self.params["parking_rate"])
        P_ = np.array(
            [0, self.params["parking_turn_P_o"]]
        )
        D_ = np.array(
            [0, self.params["parking_turn_D_o"]]
        )
        side = self.parking_stall["side"]
        initial_yaw = self.odom.pose.pose.orientation.z
        target_yaw = self.params["parking_turn_angle"][side]
        while True:
            yaw = self.odom.pose.pose.orientation.z - initial_yaw
            error = np.array([0, yaw - target_yaw])
            rospy.loginfo(f"DEBUG yaw: {yaw}")
            rospy.loginfo(f"DEBUG target_yaw: {target_yaw}")
            rospy.loginfo(f"DEBUG error: {error}")
            if abs(error[1]) < self.params["parking_turn_epsilon"]:
                break
            self.parking_pid(error, P_, D_)
            self.vel_pub.publish(self.twist)
            rate.sleep()

        self.state = DS.Stage3Parking_Reverse

    def parking_reverse_state(self):
        rate = rospy.Rate(self.params["parking_rate"])
        P_ = np.array(
            [self.params["parking_P_x"], self.params["parking_reverse_P_o"]]
        )
        D_ = np.array(
            [self.params["parking_D_x"], self.params["parking_reverse_D_o"]]
        )
        while True:
            yellow_lane_error = self.parking_yellow_lane_error()
            if self.tof_distance > self.params["parking_reverse_max_tof_distance_alignment"]:
                omega_error = 0
            elif yellow_lane_error is not None:
                omega_error = yellow_lane_error
            else:
                omega_error = 0
            distance_error = (
                self.tof_distance - self.params["parking_reverse_target_tof_distance"]
            )
            error = np.array([distance_error, omega_error])
            rospy.loginfo(f"DEBUG error: {error}")
            if abs(distance_error) < self.params["parking_reverse_epsilon"]:
                break
            self.parking_pid(error, P_, D_)
            rospy.loginfo(f"DEBUG v: {self.twist.v}")
            rospy.loginfo(f"DEBUG omega: {self.twist.omega}")
            self.vel_pub.publish(self.twist)
            rate.sleep()
        rate = rospy.Rate(1 / self.params["parking_reverse_constant_time"])

        # reverse backwards for a parameterized amount time
        self.twist.v = self.params["parking_reverse_v"]
        self.vel_pub.publish(self.twist)
        rate.sleep()
        self.twist.v = 0
        self.vel_pub.publish(self.twist)
        self.is_parked = True
        self.state = DS.ShuttingDown


if __name__ == "__main__":
    node = LaneFollowNode("lanefollow_node")
    rate = rospy.Rate(8)  # 8hz

    while not rospy.is_shutdown():
        node.evaluate_errors()
        node.state_decision()
        node.drive()
        rate.sleep()
