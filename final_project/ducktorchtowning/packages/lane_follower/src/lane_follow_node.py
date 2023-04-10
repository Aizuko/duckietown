#!/usr/bin/env python3
import rospy
import cv2
import time
import json

from duckietown.dtros import DTROS, NodeType
from dataclasses import dataclass
from enum import IntEnum, auto, unique
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
from geometry_msgs.msg import Transform, Vector3
from std_msgs.msg import Float64
from tf2_ros import Buffer, TransformListener

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
STOP_MASK = [(0, 70, 150), (20, 255, 255)]
DEBUG = True

DUCKIES_PLUS_LINE = [(0, 70, 120), (40, 255, 255)]
DUCKIES_ONLY = [(0, 55, 145), (20, 255, 255)]
REDLINE_MASK = [(0, 100, 120), (10, 255, 255)]
BLUELINE_MASK = [(40, 100, 80), (130, 255, 255)]


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
    Stage3Parking_Depth = 31
    Stage3Parking_Overshoot = 32
    Stage3Parking_Reverse = 33
    Stage3Parking_ThinkDuck = 39

    ShuttingDown = 90


class TagType(IntEnum):
    """IntEnum mirror of the classification @ apriltag tag.py"""

    RightStop = 1
    LeftStop = 2
    ForwardStop = 3
    CrossingStop = 4
    ParkingLotEnteringStop = 5


class LaneFollowNode(DTROS):
    def __init__(self, node_name):
        super(LaneFollowNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC
        )
        self.node_name = node_name
        self.bridge = CvBridge()
        self.veh = rospy.get_param("~veh")

        with open("/params.json") as f:
            self.params = json.load(f)

        self.params = {
            **self.params["default"],
            **(self.params.get(self.veh) or {}),
        }

        # ╔─────────────────────────────────────────────────────────────────────╗
        # │ Sτατε cδητrδls                                                      |
        # ╚─────────────────────────────────────────────────────────────────────╝
        self.state = DS(self.params["starting_state"])
        self.state_start_time = time.time()

        self.is_parked = False
        self.duck_free_time = 0.0
        self.last_seen_duck = None

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

        self.right_last_time = rospy.get_time()
        self.left_last_time = rospy.get_time()
        self.bottom_last_time = rospy.get_time()

        self.parking_last_error = 0
        self.parking_last_time = rospy.get_time()

        self.P = 0.049
        self.D = -0.004

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

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        # Shutdown hook
        rospy.on_shutdown(self.hook)

    def state_decision(self, most_recent_digit):
        if self.is_parked:
            self.state = DS.ShuttingDown
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("End of the road")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            rospy.signal_shutdown("Rode to the end of the road")

        elif most_recent_digit == 7 and all(
            [
                self.seen_ints[0],
                self.seen_ints[5],
                self.seen_ints[8],
                self.seen_ints[2],
                self.seen_ints[1],
            ]
        ):
            self.state = DS.ForceLeft
        elif most_recent_digit == 7:
            self.state = DS.ForceForward
        elif most_recent_digit == 6 and self.seen_ints[9] != 0:
            self.state = DS.ForceForward
        else:
            self.state = DS.LaneFollowing

    def ap_callback(self, msg):
        self.ap_distance = msg.x
        self.ap_label = TagType(int(msg.y))

        if self.ap_label == TagType.ParkingLotEnteringStop:
            self.state = DS.Stage3Drive_ThinkDuck
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

        todo("Finish this")

    def lane_callback(self, msg):
        image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

        if self.state == DS.Stage2Ducks_WaitForCrossing:
            if not self.is_good2go(image):
                return
            self.state = DS.Stage2Ducks_LaneFollowing

        right_image = image[:, 400:, :]
        left_image = image[:, :-400, :]
        bottom_image = image[300:, :, :]

        crop_width = bottom_image.shape[1]

        hsv = cv2.cvtColor(right_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        right_image = cv2.bitwise_and(right_image, right_image, mask=mask)
        right_conts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        hsv = cv2.cvtColor(left_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        left_image = cv2.bitwise_and(left_image, left_image, mask=mask)
        left_conts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        hsv = cv2.cvtColor(bottom_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        bottom_image = cv2.bitwise_and(bottom_image, bottom_image, mask=mask)
        bottom_conts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        right_areas = np.array([cv2.contourArea(a) for a in right_conts])
        left_areas = np.array([cv2.contourArea(a) for a in left_conts])
        bottom_areas = np.array([cv2.contourArea(a) for a in bottom_conts])

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

        if len(bottom_areas) == 0 or np.max(bottom_areas) < 20:
            self.bottom_error = None
        else:
            max_idx = np.argmax(bottom_areas)

            M = cv2.moments(bottom_conts[max_idx])

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            self.bottom_error = cx - int(crop_width / 2) + self.lane_offset
            # except:
            #    rospy.loginfo_throttle(2, f"bottom exception composed up")
            #    pass

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
        rospy.loginfo_throttle(2, f"State: {self.state.name}")
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
        elif self.state == DS.Stage3Parking_Depth:
            self.parking_depth_state()
        elif self.state == DS.Stage3Parking_Overshoot:
            self.parking_overshoot_state()
        elif self.state == DS.Stage3Parking_Reverse:
            self.parking_reverse_state()
        elif self.state == DS.Stage3Parking_ThinkDuck:
            self.twist.v = self.twist.omega = 0
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

        if parking_stall["depth"] == "far":
            self.parking_depth_state()
        self.parking_overshoot_state(parking_stall)
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
        v = np.clip(v, -self.params["parking_max_v"], self.params["parking_max_v"])
        omega = P[1] + D[1]
        self.twist.v = v
        self.twist.omega = omega

    def parking_depth_state(self):
        rate = rospy.Rate(self.params["parking_rate"])
        while True:
            try:
                odometry_transform = self.tf_buffer.lookup_transform(
                    "world",
                    "odometry",
                    rospy.Time(0),
                    rospy.Duration(1.0),
                ).transform
                translation = odometry_transform.translation
            except Exception:
                rate.sleep()
                continue
            error = np.array(
                [translation.x - self.params["parking_far_depth_x"], 0]
            )
            rospy.logdebug_throttle(5, (error, translation.x, translation.y))
            if np.linalg.norm(error) < self.params["parking_far_depth_epsilon"]:
                break
            self.parking_pid(error)
            self.vel_pub.publish(self.twist)
            rate.sleep()
        self.state = DS.Stage3Parking_Overshoot

    def parking_overshoot_state(self):
        rate = rospy.Rate(self.params["parking_rate"])
        if self.opposite_stall["side"] == "left":
            target_angle = self.params["parking_overshoot_angle_left"] * np.pi
        else:
            target_angle = self.params["parking_overshoot_angle_right"] * np.pi
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
            error = np.array([self.params["parking_rotating_x_error"], angle_error])
            rospy.logdebug(f"angle: {angle}")
            rospy.logdebug(f"target angle: {target_angle}")
            if np.linalg.norm(error) < self.params["parking_overshoot_angle_epsilon"] * np.pi:
                break
            self.twist.v = self.params["parking_overshoot_v"]
            self.twist.omega = np.sign(target_angle) * self.params["parking_overshoot_omega"] * np.pi
            self.vel_pub.publish(self.twist)
            rate.sleep()
        self.state = DS.Stage3Parking_Reverse

    def parking_reverse_state(self):
        if self.opposite_stall["side"] == "left":
            target_angle = self.params["parking_overshoot_angle_left"] * np.pi
        else:
            target_angle = self.params["parking_overshoot_angle_right"] * np.pi
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
            error = np.array(
                [translation.y - self.parking_stall["y_parking_target"], angle - target_angle]
            )
            rospy.logdebug_throttle(5, (error, translation.x, translation.y))
            if np.linalg.norm(error) < self.params["parking_far_depth_epsilon"]:
                break
            self.parking_pid(error)
            self.vel_pub.publish(self.twist)
            rate.sleep()
        self.state = DS.ShuttingDown


if __name__ == "__main__":
    node = LaneFollowNode("lanefollow_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        node.drive()
        rate.sleep()
