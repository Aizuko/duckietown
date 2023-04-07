#!/usr/bin/env python3
import rospy
import cv2
import time
import json

from duckietown.dtros import DTROS, NodeType
from dataclasses import dataclass
from enum import Enum, auto, unique
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
from geometry_msgs.msg import Transform
from std_msgs.msg import Float64
from tf2_ros import Buffer, TransformListener

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
STOP_MASK = [(0, 70, 150), (20, 255, 255)]
DEBUG = True


@unique
class DuckieState(Enum):
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

    Stage3Drive = 30
    Stage3Drive_ThinkDuck = 39


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
        self.state = DuckieState.Stage1Loops_LaneFollowing
        self.state_start_time = time.time()

        self.is_parked = False

        # ╔─────────────────────────────────────────────────────────────────────╗
        # │ Lαηε fδllδωiηg PID sεττiηgs                                         |
        # ╚─────────────────────────────────────────────────────────────────────╝
        self.bottom_error = None
        self.right_error = None
        self.left_error = None

        self.lane_offset = 220

        self.velocity = self.params["velocity"]
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        self.right_last_error = 0
        self.left_last_error = 0
        self.bottom_last_error = 0

        self.last_time = rospy.get_time()

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
        self.sub = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.lane_callback,
            queue_size=1,
            buff_size="20MB",
        )
        self.vel_pub = rospy.Publisher(
            f"/{self.veh}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1
        )

        self.sub_teleport = rospy.Subscriber(
            f"/{self.veh}/deadreckoning_node/teleport",
            Transform,
            self.cb_teleport,
            queue_size=1,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        # Shutdown hook
        rospy.on_shutdown(self.hook)

    def state_decision(self, most_recent_digit):
        if self.is_parked:
            self.state = DuckieState.ShuttingDown
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("ALL DIGITS HAVE BEEN SEEN")
            print("Signaling shutdown")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            rospy.signal_shutdown("All digits have been seen")
        elif most_recent_digit == 7 and all(
            [
                self.seen_ints[0],
                self.seen_ints[5],
                self.seen_ints[8],
                self.seen_ints[2],
                self.seen_ints[1],
            ]
        ):
            self.state = DuckieState.ForceLeft
        elif most_recent_digit == 7:
            self.state = DuckieState.ForceForward
        elif most_recent_digit == 6 and self.seen_ints[9] != 0:
            self.state = DuckieState.ForceForward
        else:
            self.state = DuckieState.LaneFollowing

    def cb_teleport(self, msg):
        is_close = (
            self.params["detection_dist_min"]
            < self.ap_distance
            < self.params["detection_dist_max"]
        )

    def lane_callback(self, msg):
        image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

        right_image = image[:, 400:, :]
        left_image = image[:, :-400, :]
        bottom_image = image[300:, :, :]

        crop_width = bottom_image.shape[1]

        hsv = cv2.cvtColor(right_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        right_image = cv2.bitwise_and(crop, crop, mask=mask)
        right_conts, _ = cv2.findContours(
            right_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        hsv = cv2.cvtColor(left_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        left_image = cv2.bitwise_and(crop, crop, mask=mask)
        left_conts, _ = cv2.findContours(
            left_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        hsv = cv2.cvtColor(bottom_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        bottom_image = cv2.bitwise_and(crop, crop, mask=mask)
        bottom_conts, _ = cv2.findContours(
            bottom_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        right_areas = np.array([cv2.contourArea(a) for a in right_conts])
        left_areas = np.array([cv2.contourArea(a) for a in left_conts])
        bottom_areas = np.array([cv2.contourArea(a) for a in bottom_conts])

        if len(right_areas) == 0 or np.max(right_areas) < 20:
            self.error = None
        else:
            max_idx = np.argmax(right_areas)

            M = cv2.moments(right_conts[max_idx])
            try:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                self.right_error = cx - int(crop_width / 2) + self.offset
            except:
                pass

        if len(left_areas) == 0 or np.max(left_areas) < 20:
            self.error = None
        else:
            max_idx = np.argmax(left_areas)

            M = cv2.moments(left_conts[max_idx])
            try:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                self.left_error = cx - int(crop_width / 2) + self.offset
            except:
                pass

        if len(bottom_areas) == 0 or np.max(bottom_areas) < 20:
            self.error = None
        else:
            max_idx = np.argmax(bottom_areas)

            M = cv2.moments(bottom_conts[max_idx])
            try:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                self.bottom_error = cx - int(crop_width / 2) + self.offset
            except:
                pass

    def drive(self):
        todo("Switch to using correct errors in drive")

        delta_t = time.time() - self.state_start_time
        if self.params.get("state") is not None:
            self.state = DuckieState[self.params["state"]]
        rospy.loginfo_throttle(2, f"State: {self.state.name}")

        if self.state == DuckieState.Classifying:
            self.twist.v = 0
            self.twist.omega = 0
        elif self.state == DuckieState.ShuttingDown:
            self.twist.v = 0
            self.twist.omega = 0
        elif (
            self.state == DuckieState.ForceLeft
            and delta_t < self.params["force_left_duration"]
        ):
            self.twist.v = self.params["force_left_velocity"]
            self.twist.omega = self.params["force_left_omega"]
        elif self.state == DuckieState.ForceLeft:
            self.state = DuckieState.LaneFollowing
            self.state_start_time = time.time()
            return
        elif (
            self.state == DuckieState.ForceForward
            and delta_t < self.params["force_forward_duration"]
        ):
            self.twist.v = self.params["force_forward_velocity"]
            self.twist.omega = self.params["force_forward_omega"]
        elif self.state == DuckieState.ForceForward:
            self.state = DuckieState.LaneFollowing
            self.state_start_time = time.time()
            return
        elif self.state == DuckieState.Parking:
            self.parking_state()
        elif self.error is None:
            self.twist.omega = 0
            self.twist.v = self.velocity
        else:
            # P Term
            P = -self.error * self.P

            # D Term
            d_error = (self.error - self.last_error) / (
                rospy.get_time() - self.last_time
            )
            self.last_error = self.error
            self.last_time = rospy.get_time()
            D = d_error * self.D

            self.twist.v = self.velocity
            self.twist.omega = P + D

        self.vel_pub.publish(self.twist)

    def hook(self):
        print("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0
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
            self.parking_depth_substate()
        self.parking_overshoot_substate(parking_stall)
        self.parking_reverse_substate(parking_stall, opposite_stall)
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

        self.twist.v = P[0] + D[0]
        self.twist.omega = P[1] + D[1]

    def parking_depth_substate(self):
        rate = rospy.Rate(self.params["parking_rate"])
        while True:
            try:
                at_transform = self.tf_buffer.lookup_transform(
                    "world",
                    "odometry",
                    rospy.Time(0),
                    rospy.Duration(1.0),
                ).transform
                translation = at_transform.translation
                rospy.logdebug_throttle(5, translation.x)
            except Exception:
                rate.sleep()
                continue
            error = np.array(
                [translation.x - self.params["parking_far_depth_x"], 0]
            )
            rospy.logdebug_throttle(5, (error, translation.x))
            if np.linalg.norm(error) < self.params["parking_far_depth_epsilon"]:
                break
            self.parking_pid(error)
            self.vel_pub.publish(self.twist)
            rate.sleep()

    def parking_overshoot_substate(self, opposite_stall):
        rate = rospy.Rate(self.params["parking_rate"])
        if opposite_stall["side"] == "left":
            angle = self.params["parking_overshoot_angle_left"] * np.pi
        else:
            angle = -self.params["parking_overshoot_angle_right"] * np.pi
        while True:
            try:
                at_transform = self.tf_buffer.lookup_transform(
                    "world",
                    "odometry",
                    rospy.Time(0),
                    rospy.Duration(1.0),
                ).transform
                translation = at_transform.translation
                rospy.logdebug_throttle(5, translation.x)
            except Exception:
                rate.sleep()
                continue
            error = [0, angle]
            rospy.logdebug_throttle(5, (error, translation.x))
            if abs(error) < self.params["parking_far_depth_epsilon"]:
                break
            self.parking_pid(error)
            self.vel_pub.publish(self.twist)
            rate.sleep()

    def parking_reverse_substate(self, parking_stall, opposite_stall):
        return


if __name__ == "__main__":
    node = LaneFollowNode("lanefollow_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        node.drive()
        rate.sleep()
