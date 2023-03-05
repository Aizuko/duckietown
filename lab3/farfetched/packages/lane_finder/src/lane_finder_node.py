#!/usr/bin/env python3
import os
import time
from pathlib import Path

import cv2
import cv2 as cv
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

RED = (0,0,255)
BLUE = (255,0,0)
GREEN = (0,255,0)
TEAL = (255,255,0)
WHITE = (255,255,255)
BLACK = (0,0,0)


def min_dist_in_set(points: list) -> list:
    """ Returns the two points with the minimal euclidean norm """
    if len(points) < 2:
        return None

    min_dist = float("inf")
    best = None

    for i in range(len(points)):
        for j in range(i+1, len(points)):
            a = np.array(points[i])
            b = np.array(points[j])

            if cv.norm(a-b) < min_dist:
                min_dist = cv.norm(a-b)
                best = [a, b]

    return best


class LaneFinderNode(DTROS):
    def __init__(self, node_name):
        super(LaneFinderNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.DRIVER,
        )

        self.hostname = rospy.get_param("~veh")
        self.bridge = CvBridge()
        self.raw_image = None
        self.processed_image = None
        self.horizontal_target_err = None
        self.is_processed = False

        #self.low_hsv = (21, 15, 134)
        #self.high_hsv = (66, 255, 255)
        self.low_hsv = (13, 70, 170)
        self.high_hsv = (32, 135, 255)

        self.counter = 0  # Debug counter

        # Setup publisher path ====
        self.pub_pose = rospy.Publisher(
            f"/{self.hostname}/lane_finder_node/pose",
            FarfetchedPose,
            queue_size=2,
        )

        self.pub_yellow = rospy.Publisher(
            f"/{self.hostname}/lane_finder_node/debug/yellow/compressed",
            CompressedImage,
            queue_size=1,
        )

        self.pub_raw = rospy.Publisher(
            f"/{self.hostname}/lane_finder_node/debug/raw/compressed",
            CompressedImage,
            queue_size=1,
        )

        self.img_sub = rospy.Subscriber(
            #f"/{self.hostname}/homography_publisher/image/compressed",
            f"/{self.hostname}/camera_node/image/compressed",
            CompressedImage,
            self.callback_image
        )

    def callback_image(self, message):
        self.is_processed = False
        self.raw_image = self.bridge.compressed_imgmsg_to_cv2(
            message, desired_encoding='passthrough'
        )
        self.is_processed = False

    def process(self):
        image = self.raw_image
        image = image[image.shape[0]//3:4*image.shape[0]//5, :]
        image = cv.blur(image, (5,5))

        y, x, _ = image.shape

        self.pub_raw.publish(
            self.bridge.cv2_to_compressed_imgmsg(image, dst_format="jpeg")
        )

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.low_hsv, self.high_hsv)

        image[mask == 0] = BLACK
        image[mask != 0] = WHITE

        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        conts, _ = cv2.findContours(image_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(conts) > 0:
            conts_sort = sorted(conts, key=lambda x: -cv2.contourArea(x))

            for i in range(4):
                if i >= len(conts_sort):
                    return

                c = conts_sort[i]

                M = cv2.moments(c)
                cx = int(M['m10']/(M['m00'] or 1))
                cy = int(M['m01']/(M['m00'] or 1))

                contour_center = np.array([cx, cy])

                cv2.circle(image, (cx, cy), 8,  RED, -1)

                rect = cv2.minAreaRect(c)
                box = np.intp(cv2.boxPoints(rect))
                cv2.drawContours(image, [box], 0, RED, 2)

                ext_left = tuple(c[c[:, :, 0].argmin()][0])
                ext_right = tuple(c[c[:, :, 0].argmax()][0])
                ext_top = tuple(c[c[:, :, 1].argmin()][0])
                ext_bot = tuple(c[c[:, :, 1].argmax()][0])

                cv2.drawContours(image, [c], 0, GREEN, 1)
                cv2.circle(image, ext_left, 8,  RED, -1)
                cv2.circle(image, ext_right, 8, RED, -1)
                cv2.circle(image, ext_top, 8,   RED, -1)
                cv2.circle(image, ext_bot, 8,   RED, -1)

                self.target = (cx - cy, cy)
                self.horizontal_target_err = int(self.target[0] - x//2)

                self.counter += 1
                if self.counter == 10:
                    self.counter = 0
                    rospy.loginfo(f"{self.target}, {x//2}, == {self.horizontal_target_err}")

                cv2.line(image, (cx, cy), self.target, TEAL, 3)
                cv2.line(image, (x//2, 0), (x//2, y), TEAL, 3)

                #cv2.line(image, a, b, TEAL, 3)
                break

        self.processed_image = image
        self.is_processed = True

    def run(self):
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            if self.raw_image is not None and not self.is_processed:
                self.process()

            if self.processed_image is not None and self.horizontal_target_err is not None:
                msg = self.bridge.cv2_to_compressed_imgmsg(
                    self.processed_image,
                    dst_format="jpeg",
                )
                self.pub_yellow.publish(msg)

                pmsg = FarfetchedPose()
                pmsg.horizontal_target_err = int(self.horizontal_target_err)
                pmsg.is_yellow_detected = True

                self.pub_pose.publish(pmsg)
                #rospy.loginfo("Published pose message")

            rate.sleep()

if __name__ == "__main__":
    node = LaneFinderNode(node_name="lane_finder_node")
    node.run()
    rospy.spin()
