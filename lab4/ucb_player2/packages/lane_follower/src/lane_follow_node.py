#!/usr/bin/env python3
import json
from collections import deque
from enum import Enum, auto, unique
from functools import lru_cache

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import LEDPattern, Twist2DStamped
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CompressedImage, Range
from std_msgs.msg import ColorRGBA
from tf import transformations as tr

# TODO: extact into config file for faster tuning
ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
STOP_MASK = [(0, 70, 150), (20, 255, 255)]

OFF_COLOR = ColorRGBA()
OFF_COLOR.r = OFF_COLOR.g = OFF_COLOR.b = OFF_COLOR.a = 0.0


@unique
class DuckieState(Enum):
    """States our duckiebot can visit. These modify the LaneFollowNode"""

    LaneFollowing = auto()
    Stopped = auto()
    BlindTurnLeft = auto()
    BlindTurnRight = auto()
    BlindForward = auto()
    Tracking = auto()


@unique
class LEDColor(Enum):
    Red = [1.0, 0.0, 0.0]
    Green = [0.0, 1.0, 0.0]
    Blue = [0.0, 0.0, 1.0]
    Yellow = [1.0, 1.0, 0.0]
    Teal = [0.0, 1.0, 1.0]
    Magenta = [1.0, 0.0, 1.0]
    Off = [0.0, 0.0, 0.0]


@unique
class LEDIndex(Enum):
    # 0 == front left
    # 1 == NONE
    # 2 == front right
    # 3 == back right
    # 4 == back left
    All = set(range(0, 5))
    Left = set([0, 4])
    Right = set([2, 3])
    Back = set([3, 4])
    Front = set([0, 2])
    BackLeft = set([4])
    BackRight = set([3])
    FrontLeft = set([0])
    FrontRight = set([2])


class FrozenClass(object):
    __isfrozen = False

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError(f"{self} is a frozen class")
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True


class LaneFollowNode(DTROS, FrozenClass):
    def __init__(self, node_name):
        super(LaneFollowNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC
        )

        # ╔────────────────────────────────────────────────────────────────────╗
        # │  Cδηsταητs (τδ τμηε)                                               |
        # ╚────────────────────────────────────────────────────────────────────╝
        # Utils
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")
        self.bridge = CvBridge()

        with open("/params.json") as f:
            self.params = json.load(f)

        self.params = {
            **self.params["default"],
            **(self.params.get(self.veh) or {}),
        }

        self.is_american = self.params["is_american"]
        self.is_debug = self.params["is_debug"]

        # Lane following
        self.offset = 220 * (2 * int(self.is_american) - 1)

        self.min_velocity = self.params["min_velocity"]
        self.velocity = self.params["velocity"]
        self.max_velocity = self.params["max_velocity"]
        self.twist = Twist2DStamped(v=self.velocity, omega=0)
        self.Px = self.params["Px"]
        self.Dx = self.params["Dx"]

        # Stopping
        self.stop_duration = self.params["stop_duration"]
        self.stop_immunity = self.params["stop_immunity"]
        self.tracking_distance = self.params["tracking_distance"]
        self.tracking_timeout = self.params["tracking_timeout"]
        self.stopline_area_min = self.params["stopline_area_min"]
        self.stopline_area_max = self.params["stopline_area_max"]

        # Tracking
        self.safe_distance = self.params["safe_distance"]
        self.blind_duration_forward = self.params["blind_duration_forward"]
        self.blind_duration_left = self.params["blind_duration_left"]
        self.blind_duration_right = self.params["blind_duration_right"]

        # ╔────────────────────────────────────────────────────────────────────╗
        # │ Dyηαmic ναriαblεs                                                  |
        # ╚────────────────────────────────────────────────────────────────────╝
        # State
        self.state = DuckieState.LaneFollowing

        # PID Variables
        self.error = None  # Error off target

        self.last_error = 0
        self.last_time = rospy.get_time()

        # Stopline variables
        self.stop_time = None
        self.next_blind_state = None
        self.blind_start_time = None
        self.is_stop_line = False

        # TOF
        self.tof_dist = [0.0, 0.0, 0.0]

        # Transform
        self.robot_transform_queue = deque(maxlen=self.params["deque_maxlen"])
        self.robot_transform_time = None

        self.tracking_error = None
        self.tracking_last_error = 0
        self.tracking_last_time = rospy.get_time()

        self.Pz = self.params["Pz"]
        self.Dz = self.params["Dz"]

        # Shutdown hook
        rospy.on_shutdown(self.on_shutdown)

        # ╔────────────────────────────────────────────────────────────────────╗
        # │ Pμblishεrs & Sμbscribεrs                                           |
        # ╚────────────────────────────────────────────────────────────────────╝
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
            f"/{self.veh}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1,
        )
        self.led_pub = rospy.Publisher(
            f"/{self.veh}/led_emitter_node/led_pattern",
            LEDPattern,
            queue_size=1,
        )
        self.sub = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.ajoin_callback,
            queue_size=1,
            buff_size="20MB",
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

        self._freeze()  # Now disallow any new attributes

    def ajoin_callback(self, msg):
        self.lane_callback(msg)

        if self.state is not DuckieState.Stopped:
            self.stop_callback(msg)

    def lane_callback(self, msg):
        img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        crop = img[300:-1, :, :]
        crop_width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Search for lane in front
        areas = np.array([cv2.contourArea(a) for a in contours])

        if len(areas) == 0 or np.max(areas) < 20:
            self.error = None
        else:
            max_idx = np.argmax(areas)

            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                self.error = cx - int(crop_width / 2) + self.offset
                if self.is_debug:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except Exception:
                pass

        if self.is_debug:
            rect_img_msg = self.bridge.cv2_to_compressed_imgmsg(crop)
            self.pub.publish(rect_img_msg)

    def tof_callback(self, msg):
        self.tof_dist.append(msg.range)  # Keep full backlog

    def robot_ahead_transform_callback(self, msg: TransformStamped):
        rospy.loginfo(f"Got transform callback {msg.header.stamp.to_sec()}")
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
        self.robot_transform_queue.append(T)

        y_rote = tr.euler_from_matrix(T)[1]

        if y_rote > self.params["left_rot"]:
            self.next_blind_state = DuckieState.BlindTurnLeft
        elif y_rote < self.params["right_rot"]:
            self.next_blind_state = DuckieState.BlindTurnRight
        else:
            self.next_blind_state = DuckieState.BlindForward

        rospy.loginfo_throttle(1, f"{y_rote}")
        self.robot_transform_time = msg.header.stamp.to_sec()

    def stop_callback(self, msg):
        img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        crop = img[300:-1, :, :]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, STOP_MASK[0], STOP_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        areas = np.array([cv2.contourArea(a) for a in contours])
        is_stopline = np.any(
            np.logical_and(
                self.stopline_area_min < areas, areas < self.stopline_area_max
            )
        )

        time = rospy.get_time()
        ltime = self.stop_time

        if is_stopline and self.state is DuckieState.Tracking:
            if ltime is None or time - ltime > self.params["stop_immunity_tracking"]:
                self.state = DuckieState.Stopped
                self.stop_time = time
            else:
                self.is_stop_line = True
        elif (
            is_stopline
            and self.state is DuckieState.LaneFollowing
            and (ltime is None or time - ltime > self.params["stop_immunity"])
        ):
            self.is_stop_line = True
        elif not is_stopline and self.is_stop_line:
            self.state = DuckieState.Stopped
            self.stop_time = time
            self.is_stop_line = False

        if self.is_debug:
            rect_img_msg = self.bridge.cv2_to_compressed_imgmsg(crop)
            self.pub_red.publish(rect_img_msg)

    def drive_bindly(self):
        self.last_error = self.error = 0
        self.twist.v = self.velocity

        if self.state is DuckieState.BlindForward:
            self.set_leds(LEDColor.Yellow, LEDIndex.Back)
            self.twist.omega = 0
        elif self.state is DuckieState.BlindTurnLeft:
            self.set_leds(LEDColor.Teal, LEDIndex.Back)
            self.twist.omega = self.params["rot_omega_l"] * np.pi
        elif self.state is DuckieState.BlindTurnRight:
            self.set_leds(LEDColor.Magenta, LEDIndex.Back)
            self.twist.omega = -self.params["rot_omega_r"] * np.pi
        else:
            raise Exception(f"Invalid state `{self.state}` for blind driving")

        # rospy.loginfo(f"Publishing blind movements for state {self.state.name}")
        self.vel_pub.publish(self.twist)

    def pid_x(self, p_coef=1):
        if self.error is None:
            self.twist.omega = 0
        else:
            # P Term
            if self.state is DuckieState.Tracking:
                P = -self.error * self.params["Pxt"]
            else:
                P = -self.error * self.Px

            # D Term
            d_error = (self.error - self.last_error) / (
                rospy.get_time() - self.last_time
            )
            self.last_error = self.error
            self.last_time = rospy.get_time()
            if self.state is DuckieState.Tracking:
                D = d_error * self.params["Dxt"]
            else:
                D = d_error * self.Dx

            self.twist.omega = P + D

    def pid_z(self):
        distance_to_robot_ahead = self.distance_to_robot_ahead()
        if distance_to_robot_ahead is None:
            return
        self.tracking_error = self.safe_distance - distance_to_robot_ahead

        if self.tracking_last_error is None:
            self.tracking_last_error = self.tracking_error

        Pz = -self.tracking_error * self.Pz
        d_error = self.tracking_error - self.tracking_last_error
        d_time = rospy.get_time() - self.tracking_last_time
        self.tracking_last_error = self.tracking_error
        self.tracking_last_time = rospy.get_time()
        Dz = d_error / d_time * self.Dz
        v = Pz + Dz
        v = np.sign(v) * np.clip(
            np.abs(v), self.min_velocity, self.max_velocity
        )
        self.twist.v = np.max((v, self.params["clip_velocity"]))

    def follow_lane(self):
        self.pid_x()
        self.twist.v = self.velocity

        self.vel_pub.publish(self.twist)

    def stop_wheels(self):
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)

    def track_bot(self):
        self.pid_z()
        self.pid_x()

        # rospy.loginfo(
        #    f"(v, omega) for tracking: {self.twist.v}, {self.twist.omega}"
        # )

        self.vel_pub.publish(self.twist)

    def distance_to_robot_ahead(self):
        """Distance between our robot and theirs. Always positive

        None is returned when there's no bot detected ahead
        """
        if (
            self.robot_transform_time is not None
            and rospy.get_time() - self.robot_transform_time < self.params["tofdist_fusion"]
        ):
            latest_transform = self.robot_transform_queue[-1]
            latest_translate = latest_transform[:3, 3]
            tof_dist_transformed = self.params["tof_a"] * self.tof_dist[-1] + self.params["tof_b"]
            if self.params["print_distance"]:
                rospy.loginfo(f"{np.linalg.norm(latest_translate)}, {self.tof_dist[-1]}, {tof_dist_transformed}")
            return min(
                np.linalg.norm(latest_translate),
                tof_dist_transformed
            )

        return None

    @lru_cache(maxsize=1)
    def set_leds(self, color: LEDColor, index_set: LEDIndex):
        led_msg = LEDPattern()

        on_color = ColorRGBA()
        on_color.r, on_color.g, on_color.b = color.value
        on_color.a = 1.0

        for i in range(5):
            led_msg.rgb_vals.append(
                on_color if i in index_set.value else OFF_COLOR
            )

        if not self.params["no_led"]:
            self.led_pub.publish(led_msg)

    def run(self):
        rate = rospy.Rate(self.params["run_rate"])

        while not rospy.is_shutdown():
            rospy.loginfo_throttle(1, f"STATE: {self.state}")

            if self.state is DuckieState.LaneFollowing:
                self.set_leds(LEDColor.Green, LEDIndex.Back)
                self.follow_lane()
                try:
                    if (
                        self.distance_to_robot_ahead() is not None
                        and self.distance_to_robot_ahead() <= self.tracking_distance
                    ):
                        self.state = DuckieState.Tracking
                except TypeError:
                    pass

            elif self.state is DuckieState.Stopped:
                self.set_leds(LEDColor.Red, LEDIndex.Back)
                if rospy.get_time() - self.stop_time >= self.stop_duration:
                    if self.robot_transform_time is None:
                        self.state = DuckieState.LaneFollowing
                    elif (
                        rospy.get_time() - self.robot_transform_time
                        < self.params["inter_plan_time"]
                    ):
                        self.state = self.next_blind_state
                    else:
                        self.state = DuckieState.LaneFollowing
                else:
                    self.stop_wheels()

            elif self.state in (
                DuckieState.BlindTurnLeft,
                DuckieState.BlindTurnRight,
                DuckieState.BlindForward,
            ):
                if self.state is DuckieState.BlindTurnLeft:
                    blind_duration = self.params["blind_duration_left"]
                elif self.state is DuckieState.BlindTurnRight:
                    blind_duration = self.params["blind_duration_right"]
                else:
                    blind_duration = self.params["blind_duration_forward"]

                if self.blind_start_time is None:
                    self.blind_start_time = rospy.get_time()
                elif (
                    rospy.get_time() - self.blind_start_time
                    > blind_duration
                ):
                    self.blind_start_time = None
                    self.state = DuckieState.LaneFollowing
                else:
                    self.drive_bindly()

            elif self.state is DuckieState.Tracking:
                self.set_leds(LEDColor.Blue, LEDIndex.Back)
                self.track_bot()

                is_timeout = (
                    rospy.get_time() - self.robot_transform_time
                    > self.tracking_timeout
                )

                d = self.distance_to_robot_ahead()

                if (
                    is_timeout
                    and d is None
                    or d is not None
                    and d > self.tracking_distance
                ):
                    rospy.loginfo(
                        f"Switching to tracking from dist: {self.distance_to_robot_ahead()}"
                    )
                    self.tracking_last_error = None
                    self.state = DuckieState.LaneFollowing

            else:
                raise Exception(f"Invalid state {self.state}")

            rate.sleep()

    def on_shutdown(self):
        print("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0

        self.vel_pub.publish(self.twist)
        self.set_leds(LEDColor.Off, LEDIndex.All)
        for i in range(8):
            self.vel_pub.publish(self.twist)
            self.set_leds(LEDColor.Off, LEDIndex.All)


if __name__ == "__main__":
    node = LaneFollowNode("lanefollow_node")
    rospy.on_shutdown(node.on_shutdown)
    node.run()
    rospy.spin()  # Just in case?
