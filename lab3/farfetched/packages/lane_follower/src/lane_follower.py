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
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped, Pose2DStamped
from farfetched_msgs.msg import FarfetchedPose
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from dt_apriltags import Detector


class LaneFollowerPIDNode(DTROS):
    """
    Attempts to minimize the
    """
    def __init__(self, node_name):
        super(LaneFollowerPIDNode, self).__init__(node_name=node_name,
                                     node_type=NodeType.GENERIC)

        self.hostname = rospy.get_param("~veh")

        # Init everything to None. Don't update with None
        self.curr_time = None
        self.curr_dist_err = None
        self.curr_rad_err = None

        # Hardcoded targets
        self.target_dist = 0.0
        self.target_rad = 0.0  # TODO: this might be wrong
            # TODO: tune these
        self.p_dist_step = 0.1
        self.p_rad_step = 0.1
        self.d_dist_step = 0.1
        self.d_rad_step = 0.1
        self.i_step = 0.1  # Currently unused. Probs don't need it

        # The acc stuff we use to publish
        self.p_dist = None
        self.p_rad = None
        self.d_dist = None
        self.d_rad = None
        self.i = None

        self.sub = rospy.Subscriber(
            f"/{self.hostname}/~todo",
            FarfetchedPose,
            self.pose_cb,
        )

        self.pub_move = rospy.Publisher(
            f'/{self.hostname}/car_cmd_switch_node/cmd',
            Twist2DStamped,
            queue_size=1,
            dt_topic_type=TopicType.DRIVER,
        )

        #self.pub_move = rospy.Publisher(
        #    f'/{self.hostname}/wheels_driver_node/wheels_cmd',
        #    WheelsCmdStamped,
        #    queue_size=1,
        #    dt_topic_type=TopicType.DRIVER,
        #)

    def rotational_offset(self, radians):
        """ Returns a rotational offset in [-pi, pi] with respect to the target
        """
        a = radians
        b = self.target_rad

        return (a - b + 2*pi + pi) % (2*pi) - pi

    def pose_cb(self, pose):
        """ Updates the P and D values based on the pose

        D value won't update the first time this is run. If the signs on P and
        D are the same, D is zeroed
        """
        if not pose.is_located:
            return  # Don't update anything?

        # Update the P value
        dist_err = self.target_dist - pose.lateral_offset
        rad_err = self.rotational_offset(pose.rotational_offset_rad)

        self.p_dist = self.p_dist_step * dist_err
        self.p_rad = self.p_rad_step * rad_err

        # Update the D value
        if (self.curr_time is not None and
            self.curr_dist_err is not None and
            self.curr_rad_err is not None):

            delta_t = pose.header.stamp - self.curr_time
            delta_dist_err = dist_err - self.curr_dist_err
            delta_rad_err = rad_err - self.curr_rad_err

            self.d_dist = self.d_dist_step * delta_dist_err / delta_t
            self.d_rad = self.d_rad_step * delta_rad_err / delta_t

            # Zero out D value if the sign is the same as P
            if self.d_dist/self.d_dist == self.p_dist/self.p_dist:
                self.d_dist = 0.0
            if self.d_rad/self.d_rad == self.p_rad/self.p_rad:
                self.d_rad = 0.0

        # Update errors for next D step
        self.curr_time = pose.header.stamp
        self.curr_dist_err = dist_err
        self.curr_rad_err = rad_err

    def pub_loop(self):
        while not rospy.is_shutdown():
            if self.p_dist is not None and self.p_rad is not None:
                cmd = Twist2DStamped()
                cmd.v = self.p_dist
                cmd.omega = self.p_rad
                self.pub_move.publish(cmd)


if __name__ == '__main__':
    node = LaneFollowerPIDNode(node_name='lane_follower_pid_node')

    node.pub_loop()
    rospy.spin()  # Just in case?
