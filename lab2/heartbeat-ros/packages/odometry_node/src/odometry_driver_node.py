#!/usr/bin/env python3

import os

import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import WheelsCmdStamped
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32, Float64MultiArray, Header, String

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
        right_wheel_integrated_distance (Float32)
        left_wheel_integrated_distance (Float32)
        world_kinematics (Float64Array)
    """
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(OdometryDriverNode, self).__init__(node_name=node_name, node_type=NodeType.DRIVER)

        self._length = 0.05

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
        self.sub_world_kinematics = rospy.Subscriber(
            "world_kinematics",
            Float64MultiArray,
            callback=self.world_kinematics_callback
        )

        self.kW = None

    def dist_callback(self, wheel, dist):
        m = dist.data
        self.distances[wheel] = m
        # rospy.loginfo(f"{wheel} wheel traveled {m}m, for a total of {self.distances[wheel]}")

    def world_kinematics_callback(self, message):
        self.kW = np.array(message.data)
        self.check_exit_duckietown()

    def target_to_robot_frame(self, target):
        # get target coordinate in robot frame
        t = self.kW[2]
        R = np.array([
            [np.cos(t), np.sin(t), 0],
            [-np.sin(t), np.cos(t), 0],
            [0, 0, 1]
        ])
        dKW = target - self.kW
        dkR = (R @ dKW)[[0,2]]
        dkR[1] %= np.pi
        return dkR

    def newton_method(self, kR_target, n=10, threshold=0.001):
        # use Newton's method solve for wheel displacements
        J = 1/2 * np.array([
            [1, 1],
            [-1/self._length, 1/self._length]
        ])
        d = np.zeros((2, ))
        for _ in range(n):
            kR = 1/2 * np.array([
                (d[0] + d[1]),
                (d[1] - d[0]) / (self._length),
            ])
            dkR = kR - kR_target
            if np.linalg.norm(dkR) < threshold:
                break
            d -= np.linalg.solve(J, dkR)
        return d

    def displacement_to_velocity(self, d):
        # map wheel displacements to wheel velocity
        v = 2 * d
        vmax = np.max(np.abs(v))
        if vmax > 1:
            v /= vmax
        return v

    def inverse_kinematics(self, target):
        dkR = self.target_to_robot_frame(target)
        d = self.newton_method(dkR)
        v = self.displacement_to_velocity(d)
        return v

    def run(self, rate=10):
        rate = rospy.Rate(rate)  # Measured in Hz

        stages = [
            {
                "name": f"FORWARD MOTION 1",
                "waypoints": np.linspace((0.32, 0.32, 0), (1.32, 0.32, 0), 2)
            },
            {
                "name": f"SPIN 1",
                "waypoints": np.linspace((1.32, 0.32, 0), (1.32, 0.32, np.pi/2), 2)
            }
        ]

        while self.kW is None:
            rate.sleep()

        threshold = 0.01

        self.loginfo(self.kW)
        for stage in stages:
            rospy.loginfo(f"STAGE: {stage['name']}")
            for waypoint in stage['waypoints']:
                while True:
                    v = self.inverse_kinematics(waypoint)
                    rospy.logdebug(f"    kW: {self.kW}     v: {v}")
                    if np.linalg.norm(self.kW - waypoint) < threshold:
                        break
                    self.publish_speed(v)
                    rate.sleep()
                self.loginfo(self.kW)

        rospy.loginfo("Finished movement, setting velocities to 0")

        self.publish_speed(np.zeros((2,)))
        rate.sleep()

    def publish_speed(self, v):
        cmd = WheelsCmdStamped()
        cmd.vel_left = v[0]
        cmd.vel_right = v[1]
        self.pub_move.publish(cmd)

    def check_exit_duckietown(self):
        if self.kW[0] < 0 or self.kW[1] < 0 or self.kW[0] > 2.01 or self.kW[1] > 3.27:
            rospy.loginfo("exited duckietown, yikes!")
            self.emergency_halt()

    def emergency_halt(self):
        self.publish_speed(np.zeros((2,)))
        rospy.loginfo("Sent emergency stop")
        rospy.signal_shutdown("emergency halt")

if __name__ == '__main__':
    # create the node
    node = OdometryDriverNode(node_name='odometry_driver_node')

    rospy.on_shutdown(node.emergency_halt)  # Stop on crash

    node.run()
    # keep spinning
    rospy.spin()
    rospy.loginfo("Finished driving. Ready to exit")
