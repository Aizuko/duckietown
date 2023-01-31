#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import WheelsCmdStamped
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Header

hostname = os.environ['VEHICLE_NAME']

class ManualDrivingNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(ManualDrivingNode, self).__init__(node_name=node_name, node_type=NodeType.DRIVER)

        self.pub_move = rospy.Publisher(
            #'~path_to/wheels_cmd_executed',
            '~path_to/wheels_cmd',
            WheelsCmdStamped,
            queue_size=10,
            dt_topic_type=TopicType.DRIVER,
        )

    def run(self, rate=2):
        rate = rospy.Rate(rate)  # Measured in Hz
        rospy.loginfo(f"Running at {rate}Hz")

        while not rospy.is_shutdown():
            cmd = WheelsCmdStamped()

            cmd.vel_left = 10.0
            cmd.vel_right = 10.0
            self.pub_move.publish(cmd)

            rate.sleep()  # Slow to 0 after gap

            cmd.vel_left = 0.0
            cmd.vel_right = 0.0
            self.pub_move.publish(cmd)

if __name__ == '__main__':
    # create the node
    node = ManualDrivingNode(node_name='manual_driver')
    node.run()
    # keep spinning
    rospy.spin()
