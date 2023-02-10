#!/usr/bin/env python3
from datetime import datetime
import json
import os
import time

import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import WheelsCmdStamped, Pose2DStamped
from std_msgs.msg import Float32, Float64MultiArray, Header, String
from led_controls.srv import LEDControlService
import rosbag
from pathlib import Path


FORWARD_DIST = 1.0  # Measured in meters
FORWARD_SPEED = 0.3


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
        self.hostname = os.environ['VEHICLE_NAME']
        self._radius = rospy.get_param(
            f'/{self.hostname}/kinematics_node/radius', 0.025
        )
        self._length = 0.05

        self.distances = { 'left': 0.0, 'right': 0.0 }

        try:
            with open("/params.json") as f:
                self.params = json.load(f)
        except FileNotFoundError:
            self.params = {}
        self.params = self.params.get(self.hostname, {})
        rospy.loginfo("Params: ")
        rospy.loginfo(json.dumps(self.params, indent=4))

        self.EMERGENCY_STOPPED = False

        self.pub_move = rospy.Publisher(
            f'/{self.hostname}/wheels_driver_node/wheels_cmd',
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
        LED_CONTROL_SERVICE_NAME = f'/{self.hostname}/led_controls/led_control_service'
        rospy.logdebug(f"Waiting for {LED_CONTROL_SERVICE_NAME}")
        rospy.wait_for_service(LED_CONTROL_SERVICE_NAME)
        rospy.logdebug(f"Found {LED_CONTROL_SERVICE_NAME}")
        self.switch_led = rospy.ServiceProxy(
            LED_CONTROL_SERVICE_NAME,
            LEDControlService
        )
        self.kW = None
        bag_name = datetime.now().isoformat()
        bag_name = bag_name.replace('-', '.')
        bag_name = bag_name.replace(':', '.')
        bag_filename = f'/data/bags/odometry_at_{bag_name}.bag'
        Path(bag_filename).parent.mkdir(parents=True, exist_ok=True)
        self.bag = rosbag.Bag(bag_filename, 'w')
        self.bag_closed = False
        rospy.loginfo(f"Made a bag {self.bag}")

    def dist_callback(self, wheel, dist):
        m = dist.data
        self.distances[wheel] = m
        # rospy.loginfo(f"{wheel} wheel traveled {m}m, for a total of {self.distances[wheel]}")

    def world_kinematics_callback(self, message):
        if not self.bag_closed:
            self.bag.write("world_kinematics", message)
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

    def hardcoded_turn(self, target, clockwise=True, force=0.6):
        rate = rospy.Rate(30)
        if self.params.get("turnspeed") is not None:
            force = self.params.get("turnspeed")
        v = np.array([force, -force])
        v[0] = v[0] * self.params.get("tt", {}).get("left", 1)
        v[1] = v[1] * self.params.get("tt", {}).get("right", 1)
        if not clockwise:
            v = -v
        while not rospy.is_shutdown() and not self.EMERGENCY_STOPPED:
            self.publish_speed(v)
            rospy.logdebug(f"kW: {self.kW}",)
            rate.sleep()
            if self.params.get("turn_twitch", False):
                self.publish_speed(np.zeros((2, )))
            threshold = 0.1
            if np.abs(self.kW[2] - target % (2 * np.pi)) < threshold:
                self.publish_speed(np.zeros((2, )))
                return

    def hardcoded_forward(self, target_distance, backwards=False, speed=0.5):
        rate = rospy.Rate(30)
        if self.params.get("forwardspeed") is not None:
            speed = self.params.get("forwardspeed")
        v = np.array([speed, speed])
        v[0] = v[0] * self.params.get("t", {}).get("left", 1)
        v[1] = v[1] * self.params.get("t", {}).get("right", 1)
        if backwards:
            v = -v
        threshold = 0.1
        kW0 = self.kW.copy()[:2]
        while not rospy.is_shutdown() and not self.EMERGENCY_STOPPED:
            self.publish_speed(v)
            rospy.logdebug(f"kW: {self.kW}",)
            rate.sleep()
            # self.publish_speed(np.zeros((2, )))
            distance = np.linalg.norm(self.kW[:2] - kW0)
            if np.abs(distance - target_distance) < threshold:
                self.publish_speed(np.zeros((2, )))
                return

    def hardcoded_circle(self):
        rate = rospy.Rate(30)
        v = np.array([0.7, 0.3])
        v[0] = v[0] * self.params.get("tc", {}).get("left", 1)
        v[1] = v[1] * self.params.get("tc", {}).get("right", 1)
        target_distance = 2 * np.pi * self.params.get("circle_radius", 0.4)
        kW0 = self.kW.copy()[:2]
        threshold = 0.1
        distance = 0
        while not rospy.is_shutdown() and not self.EMERGENCY_STOPPED:
            self.publish_speed(v)
            rospy.logdebug(f"kW: {self.kW}",)
            rate.sleep()
            # self.publish_speed(np.zeros((2, )))
            distance += np.linalg.norm(self.kW[:2] - kW0)
            kW0 = self.kW.copy()[:2]
            if np.abs(target_distance - distance) < threshold:
                self.publish_speed(np.zeros((2, )))
                return

    def state_1(self):
        rospy.loginfo("STATE 1: Stay still")
        self.switch_led(1., 0., 0., 1.0)
        time.sleep(5)

    def state_2(self):
        rospy.loginfo("STATE 2: C")
        self.switch_led(0., 1., 0., 1.0)
        distance = 1.25
        rospy.loginfo("TURN 1")
        self.hardcoded_turn(0, clockwise=True, force=self.params.get("initial_turn_force", 0.8))
        rospy.loginfo("FOWARD 1")
        self.hardcoded_forward(distance)
        rospy.loginfo("TURN 2")
        self.hardcoded_turn(np.pi/2, clockwise=False)
        rospy.loginfo("FOWARD 2")
        self.hardcoded_forward(distance)
        rospy.loginfo("TURN 3")
        self.hardcoded_turn(np.pi, clockwise=False)
        rospy.loginfo("FOWARD 3")
        self.hardcoded_forward(distance)

    def state_3(self):
        rospy.loginfo("STATE 3: Go Home")
        self.switch_led(0., 0., 1., 1.0)
        distance = 1.25
        rospy.loginfo("TURN 4")
        self.hardcoded_turn(np.pi/2, clockwise=True)
        rospy.loginfo("FOWARD 4")
        self.hardcoded_forward(distance, backwards=True)

    def state_4(self):
        rospy.loginfo("STATE 4: Circle")
        self.switch_led(1., 1., 0., 1.0)
        self.hardcoded_circle()
        self.switch_led(1., 0., 1., 0.5)

    def forward_backward(self, speed):
        distance = 1.25
        self.hardcoded_forward(distance, speed=speed)
        self.hardcoded_forward(distance, backwards=True, speed=speed)

    def run(self, rate=10):
        rate = rospy.Rate(rate)  # Measured in Hz

        while self.kW is None:
            rate.sleep()

        if self.params.get("tune", False):
            self.hardcoded_forward(3)
            return

        if self.params.get("forwardbackward", False):
            self.forward_backward(self.params.get("fbspeed", 1))
            return

        start_time = time.perf_counter()
        self.state_1()
        self.state_2()
        self.state_1()
        self.state_3()
        self.state_1()
        self.state_4()
        end_time = time.perf_counter()
        self.bag_closed = True
        self.bag.close()

        rospy.loginfo(f"Total execution time: {end_time - start_time:2f} s")
        rospy.loginfo(f"final location (world frame): {self.kW}")

        time.sleep(5)

    def publish_speed(self, v):
        cmd = WheelsCmdStamped()
        cmd.vel_left = v[0]
        cmd.vel_right = v[1]
        self.pub_move.publish(cmd)

    def check_exit_duckietown(self):
        LX = 0 - 0.2
        LY = 0 - 0.2
        UX = 1.82 + 0.2
        UY = 3 + 0.2
        if self.kW[0] < LX or self.kW[1] < LY or self.kW[0] > UX or self.kW[1] > UY:
            rospy.logwarn("exited duckietown, yikes!")
            # self.emergency_halt()

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
    rospy.loginfo("Finished driving. Ready to exit")
