#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import WheelsCmdStamped
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Header, Float32

FORWARD_DIST = 1.0  # Measured in meters
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
            queue_size=1,
            dt_topic_type=TopicType.DRIVER,
        )

        self.sub_right = rospy.Subscriber(
            f'right_wheel_integrated_distance',
            Float32,
            lambda dist: self.dist_callback('right', dist),
            queue_size=1,
        )
        self.sub_left = rospy.Subscriber(
            f'left_wheel_integrated_distance',
            Float32,
            lambda dist: self.dist_callback('left', dist),
            queue_size=1,
        )

    def dist_callback(self, wheel, dist):
        m = dist.data
        self.distances[wheel] = m
        rospy.loginfo(f"{wheel} wheel traveled {m}m change, for a total of {self.distances[wheel]}")

    def run(self, rate=10):
        rate = rospy.Rate(rate)  # Measured in Hz

        rospy.loginfo("Starting forward movement")

        while (self.distances['left'] < FORWARD_DIST
               and self.distances['right'] < FORWARD_DIST):
            self.publish_speed(FORWARD_SPEED)
            rate.sleep()
            #self.publish_speed(0.0)

        rospy.loginfo("Starting reverse movement")

        while (self.distances['left'] < 2*FORWARD_DIST
               or self.distances['right'] < 2*FORWARD_DIST):
            self.publish_speed(-FORWARD_SPEED)
            #rate.sleep()
            #self.publish_speed(0.0)

        rospy.loginfo("Finished movement, setting velocities to 0")

        self.publish_speed(0.0)

    def publish_speed(self, velocity: float):
        cmd = WheelsCmdStamped()
        cmd.vel_left = velocity
        cmd.vel_right = velocity

        self.pub_move.publish(cmd)

if __name__ == '__main__':
    # create the node
    node = OdometryDriverNode(node_name='odometry_driver_node')

    def emergency_halt():
        node.publish_speed(0.0)
        rospy.loginfo("Sent emergency stop")

    rospy.on_shutdown(emergency_halt)  # Stop on crash

    node.run()
    # keep spinning
    #rospy.spin()  # Probably don't need?
    rospy.loginfo("Finished driving. Ready to exit")
