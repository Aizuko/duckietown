#!/usr/bin/env python3
from typing import List

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import LEDPattern
from geometry_msgs.msg import Quaternion, Transform, TransformStamped, Vector3
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import ColorRGBA, Header

from led_controls.srv import MallardEyedentify, MallardEyedentifyResponse


class MallardEyeNode(DTROS):
    def __init__(self, node_name):
        super(MallardEyeNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC
        )

        self.hostname = rospy.get_param("~veh")
        self.bridge = CvBridge()

        self.serv = rospy.Service(
            "mallard_eyedentification", MallardEyedentify, self.identify
        )
        rospy.loginfo("Started led_control_service")
        return

    def identify(self, srv: MallardEyedentify) -> MallardEyedentifyResponse:
        pass

    def onShutdown(self):
        super(MallardEyeNode, self).onShutdown()


if __name__ == "__main__":
    camera_node = MallardEyeNode(node_name="mallard_eye_node")
    camera_node.run(1)
