#!/usr/bin/env python3
import os
import time
from pathlib import Path

import cv2
import numpy as np
import rospy
import yaml
from augmented_reality_basics import Augmenter
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import LEDPattern
from duckietown_msgs.srv import (
    ChangePattern,
    ChangePatternResponse,
    SetCustomLEDPattern,
    SetCustomLEDPatternResponse)
from sensor_msgs.msg import CompressedImage, CameraInfo
from std_msgs.msg import ColorRGBA

# In the ROS node, you just need a callback on the camera image stream that
# uses the Augmenter class to modify the input image. Therefore, implement
# a method called callback that writes the augmented image to the appropriate
# topic.


class ARBasicsNode(DTROS):
    def __init__(self, node_name):
        super(ARBasicsNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.DRIVER)

        self.hostname = rospy.get_param("~veh")
        H = self.load_homography_matrix()
        self.augmenter = Augmenter(H)
        self.bridge = CvBridge()

        # Read in yaml ====
        yaml_file = rospy.get_param("~map_file")

        with open(yaml_file, 'r') as y:
            self.cvmap = yaml.load(y, Loader=yaml.CLoader)

        # Setup publisher path ====
        yaml_basename = Path(yaml_file).stem

        self.pub = rospy.Publisher(
            f"/{self.hostname}/arbasics_node/{yaml_basename}/image/compressed",
            CompressedImage,
            queue_size=2,
        )

        self.img_sub = rospy.Subscriber(
            f"/{self.hostname}/camera_node/image/compressed",
            CompressedImage,
            self.callback_image
        )

        self.camera_info_sub = rospy.Subscriber(
            f"/{self.hostname}/camera_node/camera_info",
            CameraInfo,
            self.callback_camera_info
        )

        self.image = None
        self.raw_image = None

    def callback_image(self, message):
        """Callback for the /camera_node/image/compressed topic."""
        self.raw_image = self.bridge.compressed_imgmsg_to_cv2(
            message, desired_encoding='passthrough'
        )

    def callback_camera_info(self, message):
        """Callback for the camera_node/camera_info topic."""
        self.augmenter.from_camera_info(message)

    def load_homography_matrix(self):
        """Load homography matrix from extrinsic calibration file."""
        for filename in [self.hostname, "default"]:
            try:
                path = "/data/config/calibrations/camera_extrinsic/"
                path += f"{filename}.yaml"
                with open(path) as f:
                    data = yaml.load(f, Loader=yaml.CLoader)
                    rospy.loginfo(
                        f"Loaded camera extrinsic calibration file: {path}"
                    )
                    return np.array(data['homography']).reshape(3, 3)
            except FileNotFoundError:
                rospy.logwarn(
                    f"Camera extrinsic calibration file not found: {path}"
                )

    def on_shutdown(self):
        """Shutdown procedure.

        At shutdown, changes the LED pattern to `LIGHT_OFF`.
        """
        pass

    def run(self):
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            if self.raw_image is not None:
                self.image = self.augmenter.process_image(self.raw_image)
                self.image = self.augmenter.render_segments(
                    self.image, self.cvmap
                )
                message = self.bridge.cv2_to_compressed_imgmsg(
                    self.image, dst_format="jpeg"
                )
                ar_node.pub.publish(message)
            rate.sleep()


if __name__ == "__main__":
    ar_node = ARBasicsNode(node_name="augmented_reality_basics_node")

    ar_node.run()

    rospy.spin()
