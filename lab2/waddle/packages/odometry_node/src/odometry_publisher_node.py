#!/usr/bin/env python3
# Written by steventango
import os

import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelEncoderStamped, WheelsCmdStamped
from std_msgs.msg import Float32, Float64MultiArray

class OdometryPublisherNode(DTROS):
    """
    Records and publishes the distance both wheels have traveled

    Publishers:
        right_wheel_integrated_distance (Float32):
            Right wheel distance traveled. Starts at 0, can't decrease
        left_wheel_integrated_distance (Float32):
            Left wheel distance traveled. Starts at 0, can't decrease
        world_kinematics (Float64MultiArray):
            [xW, yW, tW], where xW is the x coordinate in the world frame,
            yW is the y coordinate in the world frame, and tW is the angle.

    Subscribers:
        /{hostname}/right_wheel_encoder_node/tick (WheelEncoderStamped):
            Cumulative tick count on the right wheel. Reverse substracts
        /{hostname}/left_wheel_encoder_node/tick (WheelEncoderStamped):
            Cumulative tick count on the left wheel. Reverse substracts
        /{hostname}/wheels_driver_node/wheels_cmd_executed (WheelsCmdStamped)
    """
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(OdometryPublisherNode, self).__init__(
            node_name=node_name, node_type=NodeType.PERCEPTION
        )

        self.hostname = os.environ['VEHICLE_NAME']

        # Get static parameters
        self._radius = rospy.get_param(
            f'/{self.hostname}/kinematics_node/radius', 0.025
        )
        self._length = 0.05

        self.wheels = {}
        self.kW = np.array([
            0.32,
            0.32,
            np.pi/2
        ])
        for wheel in ['left', 'right']:
            self.wheels[wheel] = {
                "sub_encoder_ticks": rospy.Subscriber(
                    f'/{self.hostname}/{wheel}_wheel_encoder_node/tick',
                    WheelEncoderStamped,
                    lambda msg, wheel=wheel: self.cb_encoder_data(wheel, msg)
                ),
                "pub_integrated_distance": rospy.Publisher(
                    f'{wheel}_wheel_integrated_distance',
                    Float32,
                    queue_size=1
                ),
                "distance": 0,
                "d": 0,
                "direction": 1,
                "ticks": None,
                "velocity": 0
            }
        self.sub_executed_commands = rospy.Subscriber(
            f'/{self.hostname}/wheels_driver_node/wheels_cmd_executed',
            WheelsCmdStamped,
            self.cb_executed_commands
        )
        self.pub_world_kinematics = rospy.Publisher(
            "world_kinematics",
            Float64MultiArray,
            queue_size=1
        )

    def cb_encoder_data(self, name, msg):
        """
        Update encoder distance information from ticks.
        """
        if self.wheels[name]["ticks"] is None:
            self.wheels[name]["ticks"] = msg.data
            rospy.loginfo(f"Init {name:5} wheel to {self.wheels[name]['ticks']}")
            return

        self.wheels[name]["distance"] += (
            self.wheels[name]["direction"] * 2 * np.pi * self._radius
            * (msg.data - self.wheels[name]["ticks"]) / msg.resolution
        )
        self.wheels[name]["d"] += (
            2 * np.pi * self._radius
            * (msg.data - self.wheels[name]["ticks"]) / msg.resolution
        )
        self.wheels[name]["ticks"] = msg.data

    def cb_executed_commands(self, msg):
        """
        Use the executed commands to determine the direction of travel of each wheel.
        """
        for wheel in self.wheels:
            velocity = getattr(msg, f"vel_{wheel}")
            self.wheels[wheel]["velocity"] = velocity
            self.wheels[wheel]["direction"] = 1 if velocity >= 0 else -1

    def calculate_kinematics(self):
        """
        Calculate [x, y, t] in both robot and world frames using dl/dr
        """
        dl = self.wheels["left"]["d"]
        dr = self.wheels["right"]["d"]
        dkR = 1 / 2 * np.array([
            (dl + dr),
            0,
            (dr - dl) / self._length,
        ])
        t = self.kW[2]
        R_inv = np.array([
            [np.cos(t), -np.sin(t), 0],
            [np.sin(t), np.cos(t), 0],
            [0, 0, 1]
        ])
        self.kW += R_inv @ dkR
        self.kW[2] %= 2 * np.pi
        for wheel in self.wheels.values():
            wheel["d"] = 0

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.publish_integrated_distances()
            self.calculate_kinematics()
            self.publish_kinematics()
            rate.sleep()

    def publish_integrated_distances(self):
        for name, wheel in self.wheels.items():
            """
            rospy.loginfo(
                f"Pub: {name:5} wheel direction: "
                f"{wheel['direction']}, "
                f"distance: {wheel['distance']} m"
            )
            """
            wheel["pub_integrated_distance"].publish(wheel["distance"])

    def publish_kinematics(self):
        message = Float64MultiArray(data=self.kW)
        self.pub_world_kinematics.publish(message)


if __name__ == '__main__':
    node = OdometryPublisherNode(node_name='odometry_publisher_node')
    node.run()
    rospy.spin()
