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

        self._radius = rospy.get_param(
            f'/{hostname}/kinematics_node/radius', 0.025
        )
        self._length = 0.05

        self.distances = { 'left': 0.0, 'right': 0.0 }

        self.EMERGENCY_STOPPED = False

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
        dkR[1] %= 2 * np.pi
        return dkR

    def newton_method(self, kR_target, n=10, threshold=0.001):
        # use Newton's method solve for wheel displacements
        J = self._radius/2 * np.array([
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
        MIN_VELOCITY = 0.60
        v = d.copy()
        vmax = np.max(np.abs(d))
        if vmax == 0:
            v = np.zeros((2, ))
        elif vmax < MIN_VELOCITY:
            v /= vmax
            v = MIN_VELOCITY * v
        return v

    def inverse_kinematics(self, target):
        dkR = self.target_to_robot_frame(target)
        d = self.newton_method(dkR)
        v = self.displacement_to_velocity(d)
        return v

    def hardcoded_turn(self, target, clockwise=True):
        rate = rospy.Rate(30)
        v = np.array([5/10, -5/10])
        if not clockwise:
            v = -v
        while not rospy.is_shutdown() and not self.EMERGENCY_STOPPED:
            self.publish_speed(v)
            rospy.logdebug(f"kW: {self.kW}",)
            rate.sleep()
            self.publish_speed(np.zeros((2, )))
            threshold = 0.1
            if np.abs(self.kW[2] - target % (2 * np.pi)) < threshold:
                self.publish_speed(np.zeros((2, )))
                return

    def hardcoded_forward(self, target):
        rate = rospy.Rate(30)
        v = np.array([0.5, 0.5])
        while not rospy.is_shutdown() and not self.EMERGENCY_STOPPED:
            self.publish_speed(v)
            rospy.logdebug(f"kW: {self.kW}",)
            rate.sleep()
            # self.publish_speed(np.zeros((2, )))
            threshold = 0.1
            if np.linalg.norm(self.kW[:2] - target) < threshold:
                self.publish_speed(np.zeros((2, )))
                return

    def run(self, rate=10):
        rate = rospy.Rate(rate)  # Measured in Hz

        # states = [
        #     {
        #         "name": "STATE 1: STAY STILL",
        #         "waypoints": np.array([[0.32, 0.32, np.pi/2]])
        #     },
        #     {
        #         "name": f"STATE 2A: ROTATE 1",
        #         "waypoints": np.linspace((0.32, 0.32, np.pi/2), (0.32, 0.32, 0), 2)
        #     },
        #     {
        #         "name": f"STATE 2B: FORWARD MOTION 1",
        #         "waypoints": np.linspace((0.32, 0.32, 0), (1.57, 0.32, 0), 10)
        #     },
        #     {
        #         "name": f"STATE 2C: ROTATE 2",
        #         "waypoints": np.array([[1.57, 0.32, np.pi/2]])
        #     }
        # ]

        while self.kW is None:
            rate.sleep()

        rospy.loginfo("TURN 1")
        self.hardcoded_turn(0, clockwise=True)
        rospy.loginfo("FOWARD 1")
        self.hardcoded_forward(np.array([1.57, 0.32]))
        rospy.loginfo("TURN 2")
        self.hardcoded_turn(np.pi/2, clockwise=False)
        rospy.loginfo("FOWARD 2")
        self.hardcoded_forward(np.array([1.57, 1.57]))
        rospy.loginfo("TURN 3")
        self.hardcoded_turn(np.pi, clockwise=False)
        rospy.loginfo("FOWARD 3")
        self.hardcoded_forward(np.array([0.32, 1.57]))
        rospy.loginfo("TURN 4")
        self.hardcoded_turn(3 * np.pi/2, clockwise=False)
        rospy.loginfo("FOWARD 4")
        self.hardcoded_forward(np.array([0.32, 0.32]))

        threshold = 0.05

        # self.loginfo(self.kW)
        # for stage in states:
        #     rospy.loginfo(f"STAGE: {stage['name']}")
        #     for waypoint in stage['waypoints']:
        #         rospy.logdebug(f"  waypoint: {waypoint}")
        #         while np.linalg.norm(self.kW - waypoint) > threshold:
        #             v = self.inverse_kinematics(waypoint)
        #             rospy.logdebug(f"    kW: {self.kW}     v: {v}")
        #             if rospy.is_shutdown():
        #                 break
        #             self.publish_speed(v)
        #             rate.sleep()
        #     self.loginfo(self.kW)
        #     if rospy.is_shutdown():
        #         break

        # rospy.loginfo("Finished movement, setting velocities to 0")

        # self.publish_speed(np.zeros((2,)))
        # rate.sleep()

    def publish_speed(self, v):
        cmd = WheelsCmdStamped()
        cmd.vel_left = v[0]
        cmd.vel_right = v[1]
        self.pub_move.publish(cmd)

    def check_exit_duckietown(self):
        return
        if self.kW[0] < -0.00 or self.kW[1] < -0.00 or self.kW[0] > 2.01 or self.kW[1] > 3.27:
            rospy.loginfo("exited duckietown, yikes!")
            self.emergency_halt()

    def emergency_halt(self):
        self.publish_speed(np.zeros((2,)))
        self.EMERGENCY_STOPPED = True
        rospy.loginfo("Sent emergency stop")
        rospy.loginfo(f"kW: {self.kW}",)

if __name__ == '__main__':
    # create the node
    node = OdometryDriverNode(node_name='odometry_driver_node')

    rospy.on_shutdown(node.emergency_halt)  # Stop on crash

    node.run()
    # keep spinning
    # rospy.spin()
    rospy.loginfo("Finished driving. Ready to exit")
