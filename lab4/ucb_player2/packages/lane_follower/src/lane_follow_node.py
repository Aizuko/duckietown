#!/usr/bin/env python3
import rospy
import cv2

from enum import Enum, unique
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage, Range
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
STOP_MASK = [(0, 70, 150), (20, 255, 255)]
DEBUG = True
ENGLISH = False


@unique
class DuckieState(Enum):
    """States our duckiebot can visit. These modify the LaneFollowNode"""
    LaneFollowing    = auto()
    Stopped          = auto()
    BlindTurnLeft    = auto()
    BlindTurnRight   = auto()
    BlindForward     = auto()
    Tracking         = auto()


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

        #╔─────────────────────────────────────────────────────────────────────╗
        #│  Cδηsταητs (τδ τμηε)                                                |
        #╚─────────────────────────────────────────────────────────────────────╝
        # Utils
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")
        self.bridge = CvBridge()

        # Lane following
        self.offset = 220 * (-1 if IS_ENGLISH else 1)
        self.velocity = 0.4
        self.twist = Twist2DStamped(v=self.velocity, omega=0)
        self.P = 0.049
        self.D = -0.004

        # Stopping
        self.stop_duration = 3

        #╔─────────────────────────────────────────────────────────────────────╗
        #│ Dyηαmic ναriαblεs                                                   |
        #╚─────────────────────────────────────────────────────────────────────╝
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

        # Shutdown hook
        rospy.on_shutdown(self.hook)

        #╔─────────────────────────────────────────────────────────────────────╗
        #│ Pμblishεrs & Sμbscribεrs                                            |
        #╚─────────────────────────────────────────────────────────────────────╝
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
        self.transform_sub = rospy.Subscriber(
            f"/{self.veh}/duckiebot_distance_node/transform",
            TransformStamped,
            self.tof_callback,
            queue_size=1,
        )
        self.vel_pub = rospy.Publisher(
            f"/{self.veh}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1,
        )

        # Now disallow any new attributes
        self._freeze()

    def ajoin_callback(self, msg):
        self.lane_callback(msg)

        if self.state != DuckieState.Stopped:
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
            except:
                pass

        if DEBUG:
            rect_img_msg = self.bridge.cv2_to_compressed_imgmsg(crop)
            self.pub.publish(rect_img_msg)

    def tof_callback(self, msg):
        self.tof_dist.append(msg.range)  # Keep full backlog
        self.loginfo(f"TOF: {self.tof_dist[-1]}")

    def stop_callback(self, msg):
        img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        crop = img[300:-1, :, :]
        crop_width = crop.shape[1]
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

        match state:
            case DuckieState.BlindForward:
                self.twist.omega = 0
            case DuckieState.BlindTurnLeft:
                self.twist.omega = np.pi/2
            case DuckieState.BlindTurnRight:
                self.twist.omega = -np.pi/2
            default:
                raise Exception(
                    f"Invalid state {state} for blind driving")

        self.vel_pub.publish(self.twist)

    def follow_lane(self):
        if self.error is None:
            self.twist.omega = 0
        else:
            # P Term
            P = -self.error * self.P

            # D Term
            d_error = (self.error - self.last_error) \
                / (rospy.get_time() - self.last_time)
            self.last_error = self.error
            self.last_time = rospy.get_time()
            D = d_error * self.D

            self.twist.v = self.velocity
            self.twist.omega = P + D

        self.vel_pub.publish(self.twist)

    def check_stop(self):
        delta_time = rospy.get_time() - self.stop_time

        if delta_time >= self.stop_duration:
            self.state = DuckieState.LaneFollowing
            self.todo("Choose state based on what it's observed")
        else:
            self.twist.v = 0
            self.twist.omega = 0
            self.vel_pub.publish(self.twist)

    def run(self, rate=8):
        rate = rospy.Rate(8)

        while not rospy.is_shutdown()
            match self.state:
                case DuckieState.LaneFollowing:
                    self.follow_lane()
                case DuckieState.Stopped:
                    self.check_stop()
                case (DuckieState.BlindTurnLeft
                    | DuckieState.BlindTurnRight
                    | DuckieState.BlindForward):
                    self.drive_bindly(self.state)
                case DuckieState.Tracking:
                    self.todo("Use tof to track bot ahead")
                default:
                    raise Exception("")

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
