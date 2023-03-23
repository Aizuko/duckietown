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
        self.omega = None
        self.velocity = 0.3
        self.counter = 0  # Debugging

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
        self.horizontal_err = pose.horizontal_target_err
        self.omega = -np.arctan(self.horizontal_err / self.vertical)
        self.counter += 1

        if self.counter == 10:
            self.counter = 0
            rospy.loginfo(f"0-Mega: {self.omega}")

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
                rospy.loginfo_throttle(5, "Waiting to start...")
            rate.sleep()

        self.on_shutdown()


class LaneFollowerGradient(DTROS):
    """ Bins based on how bad the current lateral positioning is """
    def __init__(self, node_name):
        super(LaneFollowerGradient, self).__init__(node_name=node_name,
                                     node_type=NodeType.GENERIC)

        self.hostname = rospy.get_param("~veh")

        self.white_x = None
        self.yellow_x = None

        self.speed_l = None
        self.speed_r = None

        self.sub = rospy.Subscriber(
            f"/{self.hostname}/lane_finder_node/pose",
            FarfetchedPose,
            self.pose_cb,
        )

        self.pub_move = rospy.Publisher(
            f'/{self.hostname}/wheels_driver_node/wheels_cmd',
            WheelsCmdStamped,
            queue_size=1,
            dt_topic_type=TopicType.DRIVER,
        )

    def pose_cb(self, pose):
        self.white_x = pose.white_x
        self.yellow_x = pose.yellow_x

        self.speed_l = 0.3
        self.speed_r = 0.3

        if self.yellow_x is None:
            pass
        elif 100 < self.yellow_x < 400:
            self.speed_l += min((self.yellow_x - 100) / 220, 0.6)

        if self.white_x is None:
            pass
        if 320 < self.white_x < 450:
            self.speed_r += min((self.white_x - 320) / 130, 0.6)

        rospy.loginfo(f"Speeds: {self.speed_l} :: {self.speed_r}")

    def on_shutdown(self):
        cmd = WheelsCmdStamped()
        cmd.vel_left = 0.0
        cmd.vel_right = 0.0

        for _ in range(10):
            self.pub_move.publish(cmd)

    def pub_loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.speed_l is not None and self.speed_r is not None:
                cmd = WheelsCmdStamped()
                cmd.vel_left = self.speed_l
                cmd.vel_right = self.speed_r
                self.pub_move.publish(cmd)
            else:
                rospy.loginfo("Waiting to start...")
            rate.sleep()

        self.on_shutdown()

class LaneFollowerBasicsNode(DTROS):
    """ Bins based on how bad the current lateral positioning is """
    def __init__(self, node_name):
        super(LaneFollowerBasicsNode, self).__init__(node_name=node_name,
                                     node_type=NodeType.GENERIC)

        self.hostname = rospy.get_param("~veh")

        self.white_x = None
        self.yellow_x = None

        self.speed_l = None
        self.speed_r = None

        self.sub = rospy.Subscriber(
            f"/{self.hostname}/lane_finder_node/pose",
            FarfetchedPose,
            self.pose_cb,
        )

        self.pub_move = rospy.Publisher(
            f'/{self.hostname}/wheels_driver_node/wheels_cmd',
            WheelsCmdStamped,
            queue_size=1,
            dt_topic_type=TopicType.DRIVER,
        )

    def pose_cb(self, pose):
        self.white_x = pose.white_x
        self.yellow_x = pose.yellow_x

        # Throw away white line data

        if self.yellow_x is None:
            self.speed_l = 0.4
            self.speed_r = 0.4
        elif 0 < self.yellow_x < 100:
            self.speed_l = 0.4
            self.speed_r = 0.5
            rospy.loginfo("Left soft")
        elif 100 < self.yellow_x < 200:
            self.speed_l = 0.4
            self.speed_r = 0.3
            rospy.loginfo("Right soft")
        elif 200 < self.yellow_x < 300:
            self.speed_l = 0.5
            self.speed_r = 0.3
            rospy.loginfo("Right medium")
        elif 300 < self.yellow_x:
            self.speed_l = 0.6
            self.speed_r = 0.2
            rospy.loginfo("Right hard")

        return

        if self.white_x is None and self.yellow_x is None:
            self.speed_l = 0.4
            self.speed_r = 0.4
            rospy.loginfo("Undetected... going forward")
        elif self.white_x is None:
            self.speed_l = 0.4
            self.speed_r = 0.0
            rospy.loginfo("Right spin")
        elif self.yellow_x is None:  # Assume it's an intersection?
            self.speed_l = 0.4
            self.speed_r = 0.4
            rospy.loginfo("Probs at intersection")
        elif self.white_x < 350:
            self.speed_l = 0.2
            self.speed_r = 0.6
            rospy.loginfo("Left hard")
        elif self.white_x < 250:
            self.speed_l = 0.2
            self.speed_r = 0.4
            rospy.loginfo("Left soft")
        elif self.yellow_x > 200:
            self.speed_l = 0.6
            self.speed_r = 0.2
            rospy.loginfo("Right hard")
        elif self.yellow_x > 100:
            self.speed_l = 0.4
            self.speed_r = 0.2
            rospy.loginfo("Right soft")
        else:
            self.speed_l = 0.4
            self.speed_r = 0.4
            rospy.loginfo("Flying smooth")

    def on_shutdown(self):
        cmd = WheelsCmdStamped()
        cmd.vel_left = 0.0
        cmd.vel_right = 0.0

        for _ in range(10):
            self.pub_move.publish(cmd)

    def pub_loop(self):
        while not rospy.is_shutdown():
            if self.speed_l is not None and self.speed_r is not None:
                cmd = WheelsCmdStamped()
                cmd.vel_left = self.speed_l
                cmd.vel_right = self.speed_r
                self.pub_move.publish(cmd)
            else:
                rospy.loginfo("Waiting to start...")

        self.on_shutdown()


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
            f"/{self.hostname}/lane_finder_node/pose",
            FarfetchedPose,
            self.pose_cb_test,
            queue_size=1
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

    #def rotational_offset(self, radians):
    #    """ Returns a rotational offset in [-pi, pi] with respect to the target
    #    """
    #    a = radians
    #    b = self.target_rad

    #    return (a - b + 2*pi + pi) % (2*pi) - pi

    def pose_cb_test(self, pose):
        pass
        #rospy.loginfo(pose)

    #def pose_cb(self, pose):
    #    """ Updates the P and D values based on the pose

    #    D value won't update the first time this is run. If the signs on P and
    #    D are the same, D is zeroed
    #    """
    #    if not pose.is_located:
    #        return  # Don't update anything?

    #    # Update the P value
    #    dist_err = self.target_dist - pose.lateral_offset
    #    rad_err = self.rotational_offset(pose.rotational_offset_rad)

    #    self.p_dist = self.p_dist_step * dist_err
    #    self.p_rad = self.p_rad_step * rad_err

    #    # Update the D value
    #    if (self.curr_time is not None and
    #        self.curr_dist_err is not None and
    #        self.curr_rad_err is not None):

    #        delta_t = pose.header.stamp - self.curr_time
    #        delta_dist_err = dist_err - self.curr_dist_err
    #        delta_rad_err = rad_err - self.curr_rad_err

    #        self.d_dist = self.d_dist_step * delta_dist_err / delta_t
    #        self.d_rad = self.d_rad_step * delta_rad_err / delta_t

    #        # Zero out D value if the sign is the same as P
    #        if self.d_dist/self.d_dist == self.p_dist/self.p_dist:
    #            self.d_dist = 0.0
    #        if self.d_rad/self.d_rad == self.p_rad/self.p_rad:
    #            self.d_rad = 0.0

    #    # Update errors for next D step
    #    self.curr_time = pose.header.stamp
    #    self.curr_dist_err = dist_err
    #    self.curr_rad_err = rad_err

    #def pub_loop(self):
    #    while not rospy.is_shutdown():
    #        if self.p_dist is not None and self.p_rad is not None:
    #            cmd = Twist2DStamped()
    #            cmd.v = self.p_dist
    #            cmd.omega = self.p_rad
    #            self.pub_move.publish(cmd)


if __name__ == '__main__':
    #node = LaneFollowerPIDNode(node_name='lane_follower_pid_node')
    #node = LaneFollowerBasicsNode(node_name='lane_follower_basics_node')
    #node = LaneFollowerGradient(node_name='lane_follower_gradient_node')
    node = LaneFollowerJasper(node_name='lane_follower_jasper_nav')

    rospy.on_shutdown(node.on_shutdown)  # Stop on crash
    node.pub_loop()
    rospy.spin()  # Just in case?
