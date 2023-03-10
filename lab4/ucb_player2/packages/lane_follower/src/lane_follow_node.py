#!/usr/bin/env python3
from collections import deque
from enum import Enum, auto, unique

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
DEBUG = True
IS_ENGLISH = False

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
    Red = [1., 0., 0.]
    Green = [0., 1., 0.]
    Blue = [0., 0., 1.]
    Yellow = [1., 1., 0.]
    Teal = [0., 1., 1.]
    Magenta = [1., 0., 1.]


@unique
class LEDIndex(Enum):
    All = set(range(0, 5))
    Left = set([0, 1])
    Right = set([3, 4])
    Back = set([1, 3])
    Front = set([0, 4])


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
            node_name=node_name, node_type=NodeType.GENERIC)

        # ╔─────────────────────────────────────────────────────────────────────╗
        # │  Cδηsταητs (τδ τμηε)                                                |
        # ╚─────────────────────────────────────────────────────────────────────╝
        # Utils
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")
        self.bridge = CvBridge()

        # Lane following
        self.offset = 220 * (-1 if IS_ENGLISH else 1)
        self.velocity = 0.4
        self.twist = Twist2DStamped(v=self.velocity, omega=0)
        self.Px = 0.049
        self.Dx = -0.004

        # Stopping
        self.stop_duration = 3
        self.tracking_distance = 0.5

        # Tracking
        self.safe_distance = 0.2

        # ╔─────────────────────────────────────────────────────────────────────╗
        # │ Dyηαmic ναriαblεs                                                   |
        # ╚─────────────────────────────────────────────────────────────────────╝
        # State
        self.state = DuckieState.LaneFollowing

        # PID Variables
        self.error = None  # Error off target

        self.last_error = 0
        self.last_time = rospy.get_time()

        # Stopline variables
        self.stop_time = None

        # TOF
        self.tof_dist = [0., 0., 0.]

        # Transform
        self.robot_transform_queue = deque(maxlen=6)
        self.robot_transform_time = None

        self.tracking_error = None
        self.tracking_last_error = 0
        self.tracking_last_time = rospy.get_time()

        self.Pz = 0.049
        self.Dz = -0.004

        # Shutdown hook
        rospy.on_shutdown(self.on_shutdown)

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
        self.vel_pub = rospy.Publisher(
            f"/{self.veh}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1,
        )
        self.led_pub = rospy.Publisher(
            f'/{self.veh}/led_emitter_node/led_pattern',
            LEDPattern,
            queue_size=1,
        )

        # Now disallow any new attributes
        self._freeze()

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
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

        # Search for lane in front
        areas = np.array([cv2.contourArea(a) for a in contours])

        if len(areas) == 0 or np.max(areas) < 20:
            self.error = None
        else:
            max_idx = np.argmax(areas)

            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                self.error = cx - int(crop_width / 2) + self.offset
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except Exception:
                pass

        if DEBUG:
            rect_img_msg = self.bridge.cv2_to_compressed_imgmsg(crop)
            self.pub.publish(rect_img_msg)

    def tof_callback(self, msg):
        self.tof_dist.append(msg.range)  # Keep full backlog

    def robot_ahead_transform_callback(self, msg: TransformStamped):
        transform = msg.transform
        self.robot_transform_time = msg.header.stamp.to_sec()
        T = tr.compose_matrix(
            translate=([
                transform.translation.x,
                transform.translation.y,
                transform.translation.z
            ]),
            angles=tr.euler_from_quaternion([
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w
            ])
        )
        rospy.loginfo_throttle(10, f"T: {msg}")
        self.robot_transform_queue.append(T)

    def stop_callback(self, msg):
        img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        crop = img[300:-1, :, :]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, STOP_MASK[0], STOP_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

        areas = np.array([cv2.contourArea(a) for a in contours])
        is_stopline = np.any(np.logical_and(1000 < areas, areas < 2000))

        time = rospy.get_time()
        ltime = self.stop_time

        if is_stopline and (ltime is None or time - ltime > 6):
            self.state = DuckieState.Stopped
            self.stop_time = time

        if DEBUG:
            rect_img_msg = self.bridge.cv2_to_compressed_imgmsg(crop)
            self.pub_red.publish(rect_img_msg)

    def drive_bindly(self, state):
        self.last_error = self.error = 0
        self.twist.v = self.velocity

        if state is DuckieState.BlindForward:
            self.set_leds(LEDColor.Green, LEDIndex.Yellow)
            self.twist.omega = 0
        elif state is DuckieState.BlindTurnLeft:
            self.set_leds(LEDColor.Green, LEDIndex.Teal)
            self.twist.omega = np.pi / 2
        elif state is DuckieState.BlindTurnRight:
            self.set_leds(LEDColor.Green, LEDIndex.Magenta)
            self.twist.omega = -np.pi / 2
        else:
            raise Exception(f"Invalid state {state} for blind driving")

        self.vel_pub.publish(self.twist)

    def pid_x(self):
        if self.error is None:
            self.twist.omega = 0
        else:
            # P Term
            P = -self.error * self.Px

            # D Term
            d_error = (self.error - self.last_error) \
                / (rospy.get_time() - self.last_time)
            self.last_error = self.error
            self.last_time = rospy.get_time()
            D = d_error * self.Dx

            self.twist.omega = P + D

    def pid_z(self):
        self.tracking_error = self.distance_to_robot_ahead() - self.safe_distance
        if self.tracking_last_error is None:
            self.tracking_last_error = self.tracking_error

        # PID z
        Pz = -self.tracking_error * self.Pz
        d_error = (self.tracking_error - self.tracking_last_error)
        d_time = rospy.get_time() - self.last_time
        self.tracking_last_error = self.tracking_error
        self.last_time = rospy.get_time()
        Dz = d_error / d_time * self.Dz
        self.twist.v = Pz + Dz

    def follow_lane(self):
        self.pid_x()
        self.twist.v = self.velocity

        self.vel_pub.publish(self.twist)

        if self.distance_to_robot_ahead() <= self.tracking_distance:
            self.state = DuckieState.Tracking

    def check_stop(self):
        delta_time = rospy.get_time() - self.stop_time

        if delta_time >= self.stop_duration:
            self.state = DuckieState.LaneFollowing
            # TODO: "Choose state based on what it's observed"
        else:
            self.twist.v = 0
            self.twist.omega = 0
            self.vel_pub.publish(self.twist)

    def tracking(self):
        self.pid_x()
        self.pid_z()

        self.vel_pub.publish(self.twist)

        if self.distance_to_robot_ahead() > self.tracking_distance:
            self.state = DuckieState.LaneFollowing
            self.tracking_last_error = None

    def distance_to_robot_ahead(self):
        distance_estimates = []
        if len(self.tof_dist):
            distance_estimates.append(self.tof_dist[-1])
        if (self.robot_transform_time and self.robot_transform_time >
                rospy.get_time() - 1):
            latest_transform = self.robot_transform_queue[-1]
            latest_translate = latest_transform[:3, 3]
            distance_estimates.append(np.linalg.norm(latest_translate))
        return min(distance_estimates)

    def set_leds(self, color: LEDColor, index_set: LEDIndex):
        led_msg = LEDPattern()

        on_color = ColorRGBA()
        on_color.r, on_color.g, on_color.b = color.value
        on_color.a = 1.0

        for i in range(5):
            led_msg.rgb_vals.append(on_color if i in index_set else OFF_COLOR)

        self.led_pub.publish(led_msg)

    def run(self, rate=8):
        rate = rospy.Rate(8)

        while not rospy.is_shutdown():
            rospy.loginfo_throttle(1, f"STATE: {self.state}")
            if self.state is DuckieState.LaneFollowing:
                self.set_leds(LEDColor.Green, LEDIndex.Back)
                self.follow_lane()
            elif self.state is DuckieState.Stopped:
                self.set_leds(LEDColor.Red, LEDIndex.Back)
                self.check_stop()
            elif self.state in (DuckieState.BlindTurnLeft, DuckieState.BlindTurnRight, DuckieState.BlindForward):
                self.drive_bindly(self.state)
            elif self.state is DuckieState.Tracking:
                self.set_leds(LEDColor.Blue, LEDIndex.Back)
                self.tracking()
            else:
                raise Exception(f"Invalid state {self.state}")

            rate.sleep()

    def on_shutdown(self):
        print("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)


if __name__ == "__main__":
    node = LaneFollowNode("lanefollow_node")
    rospy.on_shutdown(node.on_shutdown)
    node.run()
    rospy.spin()  # Just in case?
