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
from tag import TAG_ID_TO_TAG, Tag, TagType
from tf import transformations as tr
from tf2_ros import Buffer, ConnectivityException, TransformBroadcaster, TransformListener

"""

This is a template that can be used as a starting point for the CRA1 exercise.
You need to project the model file in the 'models' directory on an Number.
To help you with that, we have provided you with the Renderer class that render the obj file.

"""


class NumberNode(DTROS):
    def __init__(self, node_name):
        super(NumberNode, self).__init__(node_name=node_name,
                                           node_type=NodeType.GENERIC)

        self.hostname = rospy.get_param("~veh")
        self.bridge = CvBridge()

        self.camera_model = PinholeCameraModel()
        self.raw_image = None

        # Standard subscribers and publishers
        self.img_pub = rospy.Publisher(
            '~compressed',
            CompressedImage,
            queue_size=1
        )

        self.led_pattern_pub = rospy.Publisher(
            f'/{self.hostname}/led_emitter_node/led_pattern',
            LEDPattern,
            queue_size=1,
        )

        self.pub_teleport = rospy.Publisher(
            f"/{self.hostname}/deadreckoning_node/teleport",
            Transform,
            queue_size=1
        )

        self.compressed_sub = rospy.Subscriber(
            f'/{self.hostname}/camera_node/image/compressed',
            CompressedImage,
            self.cb_compressed,
            queue_size=1,
        )

        self.camera_info_sub = rospy.Subscriber(
            f"/{self.hostname}/camera_node/camera_info",
            CameraInfo,
            self.cb_camera_info,
            queue_size=1,
        )

        self.tf_broadcaster = TransformBroadcaster()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

    def process_image(self, raw):
        """Undistorts raw images.
        """
        rectified = np.zeros_like(raw)
        self.camera_model.rectifyImage(raw, rectified)
        return rectified

    def cb_camera_info(self, message):
        """Callback for the camera_node/camera_info topic."""
        self.camera_model.fromCameraInfo(message)

    def cb_compressed(self, compressed):
        self.raw_image = self.bridge.compressed_imgmsg_to_cv2(compressed)


    def rectify(self, image):
        """Undistorts raw images.
        """
        rectified = np.zeros_like(image)
        self.camera_model.rectifyImage(image, rectified)
        return rectified

    def run(self, rate=3):
        rate = rospy.Rate(rate)

        while not rospy.is_shutdown():
            if self.raw_image is None:
                rate.sleep()
                continue
            image = self.raw_image.copy()
            image = self.rectify(image)
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = self.detect(grayscale)
            for detection in detections:
                self.render_tag(image, detection)
            self.broadcast_transforms(detections)
            color = self.get_led_color(detections)
            message = self.bridge.cv2_to_compressed_imgmsg(
                image, dst_format="jpeg"
            )
            self.img_pub.publish(message)
            led_message = self.create_led_message(color)
            self.led_pattern_pub.publish(led_message)
            rate.sleep()

    def onShutdown(self):
        super(NumberNode, self).onShutdown()


if __name__ == '__main__':
    camera_node = NumberNode(node_name='number_node')
    camera_node.run(1)
