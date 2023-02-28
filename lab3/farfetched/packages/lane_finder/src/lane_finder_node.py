#!/usr/bin/env python3
import os
import time
from pathlib import Path

import cv2
import numpy as np
import rospy
import yaml
from lane_finder_tools import Augmenter
from cv_bridge import CvBridge
from farfetched_msgs.msg import FarfetchedPose
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import LEDPattern
from duckietown_msgs.srv import (
    ChangePattern,
    ChangePatternResponse,
    SetCustomLEDPattern,
    SetCustomLEDPatternResponse)
from sensor_msgs.msg import CompressedImage, CameraInfo
from std_msgs.msg import ColorRGBA


def rgb2bgr(r, g, b):
    return [b, g, r]

def mask_range_rgb(image, lower: list, upper: list, fill: list):
    return mask_range(image, rgb2bgr(*lower), rgb2bgr(*upper), rgb2bgr(*fill))

def mask_range(image, lower: list, upper: list, fill: list):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    image[mask > 0] = fill
    return image


class LaneFinderNode(DTROS):
    def __init__(self, node_name):
        super(LaneFinderNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.DRIVER)

        self.hostname = rospy.get_param("~veh")
        self.bridge = CvBridge()
        H = self.load_homography_matrix()
        self.augmenter = Augmenter(H)

        # Setup publisher path ====
        self.pub = rospy.Publisher(
            f"/{self.hostname}/lane_finder_node/pose",
            FarfetchedPose,
            queue_size=2,
        )

        self.img_sub = rospy.Subscriber(
            f"/{self.hostname}/homography_publisher/image/compressed",
            CompressedImage,
            self.callback_image
        )

        self.white_x = None
        self.yellow_x = None
        return


    def callback_image(self, message):
        """Callback for the /camera_node/image/compressed topic."""
        self.raw_image = self.bridge.compressed_imgmsg_to_cv2(
            message, desired_encoding='passthrough'
        )

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

    def channel_masking(self, image: np.ndarray):
        white_channel = mask_range_rgb(image.copy(), [160, 0, 0],   [255, 61, 255], [255]*3)
        red_channel = mask_range_rgb(image.copy(), [130, 100, 0], [255, 255, 20], [255]*3)
        yellow_channel = mask_range_rgb(image.copy(), [100, 40, 0], [240, 255, 80], [255]*3)

        white_channel = mask_range_rgb(white_channel, [0]*3, [254]*3, [0]*3)
        red_channel = mask_range_rgb(red_channel, [0]*3, [254]*3, [0]*3)
        yellow_channel = mask_range_rgb(yellow_channel, [0]*3, [254]*3, [0]*3)

        white_grey = cv2.cvtColor(white_channel, cv2.COLOR_BGR2GRAY)
        white_conts, _ = cv2.findContours(white_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(white_conts) > 0:
            c = max(white_conts, key=cv2.contourArea)
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])  # We don't use this
            self.white_x = cx
        else:
            self.white_x = None

        yellow_grey = cv2.cvtColor(yellow_channel, cv2.COLOR_BGR2GRAY)
        yellow_conts, _ = cv2.findContours(yellow_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(yellow_conts) > 0:
            c = max(yellow_conts, key=cv2.contourArea)
            M = cv2.moments(c)
            cx = int(M['m10']/(M['m00'] or 1))
            cy = int(M['m01']/(M['m00'] or 1))  # We don't use this
            self.yellow_x = cx
        else:
            self.yellow_x = None

        return

    def run(self):
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            if self.yellow_x is None and self.white_x is None:
                msg = FarfetchedPose()
                msg.lateral_offset = 0.0
                msg.rotational_offset_rad = 0.0
                msg.is_in_lane = False
                msg.is_located = False
                self.pub.publish(msg)
            else:
                msg = FarfetchedPose()
                msg.lateral_offset = 0.0
                msg.rotational_offset_rad = 0.0
                msg.is_in_lane = False
                msg.is_located = True
                self.pub.publish(msg)

            rate.sleep()


if __name__ == "__main__":
    ar_node = LaneFinderNode(node_name="lane_finder_node")
    ar_node.run()
    rospy.spin()
