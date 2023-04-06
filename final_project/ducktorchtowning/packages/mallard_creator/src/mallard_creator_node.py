#!/usr/bin/env python3
from typing import List

import cv2
import cv2 as cv
import numpy as np
import rospy
import time
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from geometry_msgs.msg import Quaternion, Transform, TransformStamped, Vector3
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import ColorRGBA, Header


class MallardCreateNode(DTROS):
    def __init__(self, node_name):
        super(MallardCreateNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC
        )

        self.hostname = rospy.get_param("~veh")
        self.bridge = CvBridge()

        self.last_time = time.time()

        self.sub_comp = rospy.Subscriber(
            f'/{self.hostname}/camera_node/image/compressed',
            CompressedImage,
            self.callback_image,
            queue_size=1,
        )

        rospy.loginfo("Started recording")

    def callback_image(self, compressed):
        if time.time() - self.last_time > 0.2:
            self.raw_image = self.bridge.compressed_imgmsg_to_cv2(compressed)
            p = cv.imwrite(f'/data/custom_data/{int(time.time())}.jpeg', self.raw_image)
            self.last_time = time.time()
            rospy.loginfo(f"wrote an image, with return val {p}")


if __name__ == "__main__":
    camera_node = MallardCreateNode(node_name="mallard_eye_node")
    rospy.spin()
