#!/usr/bin/env python3

import rospy
import cv2
import time
import json

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
from geometry_msgs.msg import Transform
from std_msgs.msg import Float64
from mallard_eye.srv import MallardEyedentify, MallardEyedentifyResponse

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
STOP_MASK = [(0, 70, 150), (20, 255, 255)]
DEBUG = True
ENGLISH = False

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

        self.seen_ints = [0 for _ in range(10)]

        self.ap_distance = 1000
        self.started_service_call = False

        # Publishers & Subscribers
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

        self.classify = rospy.ServiceProxy(
            f"/{self.veh}/mallard_eyedentification",
            MallardEyedentify,
        )

        self.sub_dist = rospy.Subscriber(
            f"/{self.veh}/deadreckoning_node/ap_distance",
            Float64,
            self.cb_distance,
            queue_size=1,
        )
        self.sub_teleport = rospy.Subscriber(
            f"/{self.veh}/deadreckoning_node/teleport",
            Transform,
            self.cb_teleport,
            queue_size=1,
        )

        self.loginfo("Lane follower is initialized")

        # PID Variables
        self.error = None
        if ENGLISH:
            self.offset = -220
        else:
            self.offset = 220
        self.velocity = self.params["velocity"]
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        self.last_error = 0
        self.last_time = rospy.get_time()

        # Stopline variables
        self.is_stopped = False
        self.stop_time = None

        # Constants
        self.P = 0.049
        self.D = -0.004
        self.stop_duration = 1

        # Shutdown hook
        rospy.on_shutdown(self.hook)

    def cb_distance(self, msg):
        self.ap_distance = msg.data

    def cb_teleport(self, msg):
        is_close = self.params["detection_dist_min"] < self.ap_distance < self.params["detection_dist_max"]

        is_stop_immune = self.stop_time and time.time() - self.stop_time < self.params["stop_time_duration"]

        if is_close and not is_stop_immune:
            self.is_stopped = True
            self.stop_time = time.time()

            for _ in range(8):
                self.drive()
            time.sleep(1)

            nb_class = self.classify(1).digit

            print("==================")
            if nb_class < 0:
                print(f"That one wasn't quite clear enough... Ecode {nb_class}")
            else:
                print(f"Found {nb_class}!")
                self.seen_ints[nb_class] += 1
                print(self.seen_ints)
            print("==================")

            self.is_stopped = False

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
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass

        if DEBUG:
            rect_img_msg = self.bridge.cv2_to_compressed_imgmsg(crop)
            self.pub.publish(rect_img_msg)

    def drive(self):
        if self.is_stopped:
            self.twist.v = 0
            self.twist.omega = 0
        elif self.error is None:
            self.twist.omega = 0
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


if __name__ == "__main__":
    node = LaneFollowNode("lanefollow_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        node.drive()
        rate.sleep()
