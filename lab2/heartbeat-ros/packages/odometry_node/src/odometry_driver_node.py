#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import WheelsCmdStamped
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Header, Float32

FORWARD_DIST = 25  # Measured in centimeters
FORWARD_SPEED = 0.3

hostname = os.environ['VEHICLE_NAME']

class OdometryDriverNode(DTROS):
    """
    Drives the bot forward FORWARD_DIST amount, then in reverse for the same
    distance at velocity FORWARD_SPEED

    Publishers:
        /{hostname}/wheels_driver_node/wheels_cmd (WheelsCmdStamped):
            Tells wheels to move at a certain velocity. Default max is 3

    Subscribers:
        ~right_wheel_integrated_distance (Float32)
        ~left_wheel_integrated_distance (Float32)
    """
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(OdometryDriverNode, self).__init__(node_name=node_name, node_type=NodeType.DRIVER)

        self.distances = { 'left': 0.0, 'right': 0.0 }

        self.pub_move = rospy.Publisher(
            f'/{hostname}/wheels_driver_node/wheels_cmd',
            WheelsCmdStamped,
            queue_size=10,
            dt_topic_type=TopicType.DRIVER,
        )

        self.sub_right = rospy.Subscriber(
            f'~right_wheel_integrated_distance',
            Float32,
            lambda dist: self.dist_callback('right', dist)
        )
        self.sub_left = rospy.Subscriber(
            f'~left_wheel_integrated_distance',
            Float32,
            lambda dist: self.dist_callback('left', dist)
        )

    def dist_callback(self, wheel, dist):
        self.distances[wheel] += dist
        rospy.loginfo(f"{wheel} wheel has traveled {dist}cm")

    def run(self, rate=0.5):
        rate = rospy.Rate(rate)  # Measured in Hz

        rospy.loginfo("Starting forward movement")

        while (self.distances['left'] < FORWARD_DIST
               or self.distances['right'] < FORWARD_DIST):
            self.publish_speed(FORWARD_SPEED)
            rate.sleep()
            self.publish_speed(0.0)

        rospy.loginfo("Starting reverse movement")

        while (self.distances['left'] < 2*FORWARD_DIST
               or self.distances['right'] < 2*FORWARD_DIST):
            self.publish_speed(-FORWARD_SPEED)
            rate.sleep()
            self.publish_speed(0.0)

        rospy.loginfo("Finished movement, setting velocities to 0")

        self.publish_speed(0.0)

    def publish_speed(velocity: float):
        cmd = WheelsCmdStamped()
        cmd.vel_left = velocity
        cmd.vel_right = velocity

        self.pub_move.publish(cmd)

if __name__ == '__main__':
    # create the node
    node = OdometryDriverNode(node_name='odometry_driver_node')
    node.run()
    # keep spinning
    rospy.spin()
