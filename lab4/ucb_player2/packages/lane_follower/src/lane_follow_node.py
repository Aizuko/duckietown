#!/usr/bin/env python3
import rospy
import cv2

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
STOP_MASK = [(0, 70, 150), (20, 255, 255)]
DEBUG = True
ENGLISH = False

class LaneFollowNode(DTROS):
    def __init__(self, node_name):
        super(LaneFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")

        # Publishers & Subscribers
        self.pub = rospy.Publisher(f"/{self.veh}/output/image/mask/compressed",
                                   CompressedImage,
                                   queue_size=1)
        self.pub_red = rospy.Publisher(
                                   f"/{self.veh}/output/image/red/compressed",
                                   CompressedImage,
                                   queue_size=1)
        self.sub = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed",
                                    CompressedImage,
                                    self.ajoin_callback,
                                    queue_size=1,
                                    buff_size="20MB")
        self.vel_pub = rospy.Publisher(f"/{self.veh}/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=1)

        self.bridge = CvBridge()

        self.loginfo("Initialized")

        # PID Variables
        self.error = None
        if ENGLISH:
            self.offset = -220
        else:
            self.offset = 220
        self.velocity = 0.4
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        self.last_error = 0
        self.last_time = rospy.get_time()

        # Stopline variables
        self.is_stopped = False
        self.stop_time = None
        self.last_stop_time = None

        # Constants
        self.P = 0.049
        self.D = -0.004
        self.stop_duration = 3

        # Shutdown hook
        rospy.on_shutdown(self.hook)

    def ajoin_callback(self, msg):
        self.lane_callback(msg)

        if not self.is_stopped:
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
        ltime = self.last_stop_time

        if is_stopline and (ltime is None or time - ltime > 6):
            self.is_stopped = True
            self.last_stop_time = time

        if DEBUG:
            rect_img_msg = self.bridge.cv2_to_compressed_imgmsg(crop)
            self.pub_red.publish(rect_img_msg)

    def drive(self):
        if self.error is None:
            self.twist.omega = 0
        elif self.is_stopped:
            if self.last_stop_time is None:
                print("It shouldn't be none...")
            elif rospy.get_time() - self.last_stop_time >= self.stop_duration:
                self.is_stopped = False
            self.twist.v = 0
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
