#!/usr/bin/env python3
import numpy as np
import os
import math
import cv2

import rospy
import yaml
import sys
from numpy import pi
from duckietown.dtros import DTROS, NodeType
from farfetched_msgs.msg import FarfetchedPose
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped, Pose2DStamped
from duckietown.dtros import DTROS, NodeType, TopicType
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from dt_apriltags import Detector


class LaneFollowerJasper(DTROS):
    """ Follows based on Jasper's line """
    def __init__(self, node_name):
        super(LaneFollowerJasper, self).__init__(node_name=node_name,
                                     node_type=NodeType.GENERIC)

        self.hostname = rospy.get_param("~veh")

        self.vertical = 40
        self.horizontal_err = None
        self.msg_time = None
        self.omega = None
        self.velocity = 0.6
        self.counter = 0  # Debugging

        self.P_coef = 1.0
        self.D_coef = 0.02

        self.sub = rospy.Subscriber(
            f"/{self.hostname}/lane_finder_node/pose",
            FarfetchedPose,
            self.pose_cb,
        )

        self.pub_move = rospy.Publisher(
            f'/{self.hostname}/car_cmd_switch_node/cmd',
            Twist2DStamped,
            queue_size=1,
            dt_topic_type=TopicType.DRIVER,
        )

    def pose_cb(self, pose):
        prev_msg_time = self.msg_time or pose.header.stamp
        prev_omega = self.omega or 0.0

        curr_horizontal_err = pose.horizontal_target_err
        curr_msg_time = pose.header.stamp

        P = -np.arctan(curr_horizontal_err / self.vertical)
        delta_t = (curr_msg_time - prev_msg_time).nsecs or 1.0
        D = (P - prev_omega) / delta_t

        if D >= P:  # Prevent derivative kick
            D = 0.0

        self.msg_time = curr_msg_time
        self.horizontal_err = pose.horizontal_target_err

        self.omega = self.P_coef * P - np.sign(P) * self.D_coef * D

        # Slow down debugging prints
        self.counter += 1
        if self.counter == 10:
            self.counter = 0
            rospy.loginfo(f"0-Mega: {self.omega}")
            rospy.loginfo(f"P: {P}, D: {D}")

    def on_shutdown(self):
        cmd = Twist2DStamped()
        cmd.v = 0.0
        cmd.omega = 0.0

        for _ in range(10):
            self.pub_move.publish(cmd)

    def pub_loop(self):
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            if self.omega is not None:
                cmd = Twist2DStamped()
                cmd.v = self.velocity
                cmd.omega = np.sign(self.omega) * min(np.pi - 0.02, np.abs(self.omega*2))

                self.pub_move.publish(cmd)
            else:
                rospy.loginfo("Waiting to start...")
            rate.sleep()

        self.on_shutdown()


if __name__ == '__main__':
    node = LaneFollowerJasper(node_name='lane_follower_jasper_nav')

    rospy.on_shutdown(node.on_shutdown)  # Stop on crash
    node.pub_loop()
    rospy.spin()  # Just in case?
