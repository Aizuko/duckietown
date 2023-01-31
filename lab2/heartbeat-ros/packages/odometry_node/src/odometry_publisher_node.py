#!/usr/bin/env python3
# Written by steventango
import os

import numpy as np
import rosbag
import rospy
import time
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelEncoderStamped, WheelsCmdStamped
from std_msgs.msg import Float32

hostname = os.environ['VEHICLE_NAME']

class OdometryPublisherNode(DTROS):
    """
    Records and publishes the distance both wheels have traveled

    Publishers:
        ~right_wheel_integrated_distance (Float32):
            Right wheel distance traveled. Starts at 0, can't decrease
        ~left_wheel_integrated_distance (Float32):
            Left wheel distance traveled. Starts at 0, can't decrease

    Subscribers:
        /{hostname}/right_wheel_encoder_node/tick (WheelEncoderStamped):
            Cumulative tick count on the right wheel. Reverse substracts
        /{hostname}/left_wheel_encoder_node/tick (WheelEncoderStamped):
            Cumulative tick count on the left wheel. Reverse substracts
    """

    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(OdometryPublisherNode, self).__init__(
            node_name=node_name, node_type=NodeType.PERCEPTION
        )

        # Get static parameters
        self._radius = rospy.get_param(
            f'/{hostname}/kinematics_node/radius', 100
        )

        # Subscribing to the wheel encoders
        self.sub_encoder_ticks = {}
        self.pub_integrated_distance = {}
        self.wheels = {}
        for wheel in ["left", "right"]:
            self.wheels[wheel] = {
                "sub_encoder_ticks": rospy.Subscriber(
                    f'/{hostname}/{wheel}_wheel_encoder_node/tick',
                    WheelEncoderStamped,
                    lambda msg, wheel=wheel: self.cb_encoder_data(wheel, msg)
                ),
                "pub_integrated_distance": rospy.Publisher(
                    f'~{wheel}_wheel_integrated_distance',
                    Float32,
                    queue_size=10
                ),
                "distance": 0,
                "direction": 1,
                "ticks": 0,
                "velocity": 0
            }
        self.sub_executed_commands = rospy.Subscriber(
            f'/{hostname}/wheels_driver_node/wheels_cmd_executed',
            WheelsCmdStamped,
            self.cb_executed_commands
        )

        bag_name = time.ctime().replace(' ', '_').replace(':', '-')
        self.bag = rosbag.Bag(f'/data/bags/odometry_at_{bag_name}.bag', 'w')

    def cb_encoder_data(self, wheel, msg):
        """
        Update encoder distance information from ticks.
        """
        self.bag.write(f'/{hostname}/{wheel}_wheel_encoder/tick', msg)

        self.wheels[wheel]["distance"] += (
            self.wheels[wheel]["direction"] * 2 * np.pi * self._radius
            * (msg.data - self.wheels[wheel]["ticks"]) / msg.resolution
        )
        self.wheels[wheel]["ticks"] = msg.data
        self.wheels[wheel]["pub_integrated_distance"].publish(self.wheels[wheel]["distance"])

    def cb_executed_commands(self, msg):
        """
        Use the executed commands to determine the direction of travel of each wheel.
        """
        self.bag.write(f'/{hostname}/wheels_cmd_executed', msg)
        for wheel in self.wheels:
            velocity = getattr(msg, f"vel_{wheel}")
            self.wheels[wheel]["velocity"] = velocity
            self.wheels[wheel]["direction"] = 1 if velocity > 0 else -1
            print(f"{wheel:5} wheel direction: {self.wheels[wheel]['direction']}")
        print(f"{wheel:5} wheel distance: {self.wheels[wheel]['distance']} m")


if __name__ == '__main__':
    node = OdometryPublisherNode(node_name='odometry_publisher_node')
    rospy.spin()
    node.bag.close()
