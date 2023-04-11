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
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
from geometry_msgs.msg import Transform, Vector3
from std_msgs.msg import Float64
from tf2_ros import Buffer, TransformListener
from tf import transformations as tr

ROAD_MASK = [(10, 60, 165), (40, 255, 255)]
DUCKIES_PLUS_LINE = [(0, 70, 120), (40, 255, 255)]
DUCKIES_ONLY = [(0, 55, 145), (20, 255, 255)]
REDLINE_MASK = [(0, 100, 120), (10, 255, 255)]
BLUELINE_MASK = [(40, 100, 80), (130, 255, 255)]
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
    Stage2Ducks_ThinkDuck = 29

    Stage3Parking = 30
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

        self.P = 0.049
        self.D = -0.004

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
        self.tof_sub = rospy.Subscriber(
            f"/{self.veh}/front_center_tof_driver_node/range",
            Range,
            self.tof_callback,
            queue_size=1,
        )

        self.timer = rospy.Timer(rospy.Duration(1), self.debug_callback)

        # Shutdown hook
        rospy.on_shutdown(self.hook)

    def debug_callback(self, _):
        if not self.params["is_debug"]:
            return
        rospy.loginfo(f"April tags: {len(self.seen_ap)}")

        if self.seen_ap[-1] is not None:
            rospy.loginfo(f"latest tag tag        {self.seen_ap[-1].tag.name}")
            rospy.loginfo(f"Delta from latest tag {time.time() - self.seen_ap[-1].time}")
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

        for i in range(-1, -(10**6), -1):
            try:
                ap = self.seen_ap[i]

                if ap is None:
                    break

                rospy.logwarn(f"==== Ap type: {ap.tag.name}")

                if not ap.is_within_time():
                    rospy.logwarn(f"Broken on time delta: {time.time() - ap.time}")
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

        return
        if self.ap_label == TagType.ParkingLotEnteringStop:
            self.state = DS.Stage3Parking_ThinkDuck
            self.state_start_time = time.time()
        elif self.ap_label == TagType.CrossingStop:
            self.state = DS.Stage2Ducks_WaitForCrossing
            self.state_start_time = time.time()
        elif self.ap_label == TagType.RightStop:
            self.state = DS.Stage1Loops_ForceRight
            self.state_start_time = time.time()
        elif self.ap_label == TagType.LeftStop:
            self.state = DS.Stage1Loops_ForceLeft
            self.state_start_time = time.time()
        elif self.ap_label == TagType.ForwardStop:
            self.state = DS.Stage1Loops_ForceForward
            self.state_start_time = time.time()

    def tof_callback(self, msg):
        self.tof_distance = min(msg.range, self.params["max_tof_distance"])

    def lane_callback(self, msg):
        self.image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

    def evaluate_errors(self):
        if self.image is None or 30 <= self.state < 40:
            return

        image = self.image

        if self.state == DS.Stage2Ducks_WaitForCrossing:
            if not self.is_good2go(image):
                return
            self.state = DS.Stage2Ducks_LaneFollowing

        right_image = image[:, 400:, :]
        left_image = image[:, :-400, :]
        bottom_image = image[300:, :, :]
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
        rospy.loginfo_throttle(
            2,
            f"Errors: {self.left_error}, {self.right_error}, {self.bottom_error}",
        )

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
        elif self.state == DS.Stage2Ducks_ThinkDuck:
            self.twist.v = self.twist.omega = 0
        elif self.state == DS.Stage3Parking_ThinkDuck:
            self.twist.v = self.twist.omega = 0
            self.parking_state()
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

    def parking_state(self):
        parking_lot = self.params["parking_lot"]
        parking_stall_number = self.params["parking_stall_number"]
        parking_stall = parking_lot[parking_stall_number - 1]
        opposite_stall_number = parking_stall["opposite_stall_number"]
        opposite_stall = parking_lot[opposite_stall_number - 1]
        self.parking_turn_state(parking_stall)
        self.parking_reverse_state(parking_stall, opposite_stall)
        self.is_parked = True

    def parking_pid(self, error):
        P = error * np.array(
            [self.params["parking_P_x"], self.params["parking_P_o"]]
        )
        d_error = error - self.parking_last_error / (
            rospy.get_time() - self.parking_last_time
        )
        self.parking_last_error = error
        self.parking_last_time = rospy.get_time()
        D = d_error * np.array(
            [self.params["parking_D_x"], self.params["parking_D_o"]]
        )
        v = P[0] + D[0]
        v = np.clip(
            v, -self.params["parking_max_v"], self.params["parking_max_v"]
        )
        omega = P[1] + D[1]
        self.twist.v = v
        self.twist.omega = omega

    def parking_turn_state(self):
        rate = rospy.Rate(self.params["parking_rate"])
        side = self.opposite_stall["side"]
        depth = self.opposite_stall["depth"]
        target_angle = self.params["parking_turn_angle"][side] * np.pi
        while True:
            try:
                odometry_transform = self.tf_buffer.lookup_transform(
                    "world",
                    "odometry",
                    rospy.Time(0),
                    rospy.Duration(1.0),
                ).transform
                translation = odometry_transform.translation
                rotation = odometry_transform.rotation
                angle = tr.euler_from_quaternion(
                    [rotation.x, rotation.y, rotation.z, rotation.w]
                )[2]
            except Exception:
                rate.sleep()
                continue
            angle_error = angle - target_angle
            error = np.array([0, angle_error])
            rospy.logdebug(f"angle: {angle}")
            rospy.logdebug(f"target angle: {target_angle}")
            if (
                np.linalg.norm(error)
                < self.params["parking_turn_angle_epsilon"] * np.pi
            ):
                break
            self.twist.v = self.params["parking_turn_v"]
            self.twist.omega = (
                np.sign(target_angle)
                * self.params["parking_turn_omega"][side][depth]
                * np.pi
            )
            self.vel_pub.publish(self.twist)
            rate.sleep()
        self.state = DS.Stage3Parking_Reverse

    def parking_reverse_state(self):
        rate = rospy.Rate(self.params["parking_rate"])
        while True:
            error = np.array(
                [
                    self.tof_distance - self.params["parking_tof_distance"],
                    0,
                ]
            )
            rospy.logdebug_throttle(1, (f"error: {error}"))
            rospy.logdebug_throttle(1, (f"tof_distance: {self.tof_distance}, target: {self.params['parking_tof_distance']}"))
            if np.linalg.norm(error, 2) < self.params["parking_reverse_epsilon"]:
                break
            self.parking_pid(error)
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