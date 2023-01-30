#!/usr/bin/env python3
import numpy as np
import os
import time
import rospy
import rosbag
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, WheelsCmdStamped
from std_msgs.msg import Header, Float32, Int32, String

hostname = os.environ['VEHICLE_NAME']

bag_name = time.ctime().replace(' ', '_').replace(':', '-')
bag = rosbag.Bag(f'/data/bags/{bag_name}.bag', 'w')

class OdometryNode(DTROS):
    def __init__(self, node_name):
        """Wheel Encoder Node
        This implements basic functionality with the wheel encoders.
        """
        # Initialize the DTROS parent class
        super(OdometryNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # Get static parameters
        self._radius = rospy.get_param(f'/{hostname}/kinematics_node/radius', 100)

        # Subscribing to the wheel encoders
        self.sub_encoder_ticks_left = rospy.Subscriber(
            f'/{hostname}/left_wheel_encoder_node/tick',
            WheelEncoderStamped,
            lambda x: self.cb_encoder_data('left', x),
        )
        self.sub_encoder_ticks_right = rospy.Subscriber(
            f'/{hostname}/right_wheel_encoder_node/tick',
            WheelEncoderStamped,
            lambda x: self.cb_encoder_data('right', x),
        )
        self.sub_executed_commands = rospy.Subscriber(
            f'/{hostname}/wheels_driver_node/wheels_cmd_executed',
            WheelEncoderStamped,
            self.cb_executed_commands,
        )

        # Publishers
        self.pub_integrated_distance_left = rospy.Publisher(
            '~distance_left',
            Twist2DStamped,
            queue_size=10,
        )
        self.pub_integrated_distance_right = rospy.Publisher(
            '~distance_right',
            Twist2DStamped,
            queue_size=10,
        )

        rospy.loginfo("Initialized")

    def cb_encoder_data(self, wheel, msg):
        """ Update encoder distance information from ticks.
        """
        rospy.loginfo(f"Reading for {wheel} wheel")
        bag.write(f'/{hostname}/{wheel}_wheel_encoder_node/tick', msg)

    def cb_executed_commands(self, msg):
        """ Use the executed commands to determine the direction of travel of each wheel.
        """
        pass

if __name__ == '__main__':
    node = OdometryNode(node_name='custom_wheel_node')
    # Keep it spinning to keep the node alive
    rospy.loginfo("Starting wheel encoders")
    rospy.spin()
    rospy.loginfo("wheel_encoder_node is up and running...")
