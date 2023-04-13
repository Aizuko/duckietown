#!/usr/bin/env python3
import rospy
import time
import json

from duckietown.dtros import DTROS, NodeType
from enum import IntEnum, unique
from sensor_msgs.msg import Range
from lane_follower.srv import StartParking, StartParkingResponse
from duckietown_msgs.msg import Twist2DStamped
import numpy as np


@unique
class ParkingState(IntEnum):
    """
    Stat emachine for parking
    """

    Waiting = 0
    Stop = 30
    Forward = 31
    Turn = 32
    Reverse = 33
    ShutDown = 90


class ParkingNode(DTROS):
    def __init__(self, node_name):
        super(ParkingNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC
        )
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")

        with open("/params.json") as f:
            self.params = json.load(f)["default"]

        # ╔─────────────────────────────────────────────────────────────────────╗
        # │ Sτατε cδητrδls                                                      |
        # ╚─────────────────────────────────────────────────────────────────────╝
        self.state = ParkingState(self.params["parking_starting_state"])
        self.state_start_time = time.time()

        self.tof_distance = np.inf
        self.twist = Twist2DStamped(v=0, omega=0)

        # ╔─────────────────────────────────────────────────────────────────────╗
        # │ Pαrkiηg αττribμτεs                                                  |
        # ╚─────────────────────────────────────────────────────────────────────╝
        self.parking_last_error = 0
        self.parking_last_time = rospy.get_time()

        parking_lot = self.params["parking_lot"]
        parking_stall_number = self.params["parking_stall_number"]
        self.parking_stall = parking_lot[parking_stall_number - 1]

        # ╔─────────────────────────────────────────────────────────────────────╗
        # │ Pμblishεrs & Sμbscribεrs                                            |
        # ╚─────────────────────────────────────────────────────────────────────╝
        self.vel_pub = rospy.Publisher(
            f"/{self.veh}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1
        )
        self.tof_sub = rospy.Subscriber(
            f"/{self.veh}/front_center_tof_driver_node/range",
            Range,
            self.tof_callback,
            queue_size=1,
        )
        self.start_service = rospy.Service(
            "~start", StartParking, self.start_callback
        )

        self.timer = rospy.Timer(rospy.Duration(1), self.debug_callback)

        rospy.on_shutdown(self.shutdown_hook)

    def debug_callback(self, _):
        if not self.params["is_debug"]:
            return

    def tof_callback(self, msg):
        self.tof_distance = min(msg.range, self.params["max_tof_distance"])

    def start_callback(self, _):
        self.state = ParkingState.Stop
        return 1

    def run(self):
        rospy.loginfo_throttle(1, f"State: {self.state.name}")
        rate = rospy.Rate(self.params["parking_rate"])
        while not rospy.is_shutdown():
            if self.state == ParkingState.Stop:
                self.parking_stop_state()
            elif self.state == ParkingState.Forward:
                self.parking_forward_state()
            elif self.state == ParkingState.Turn:
                self.parking_turn_state()
            elif self.state == ParkingState.Reverse:
                self.parking_reverse_state()
            elif self.state == ParkingState.ShutDown:
                rospy.signal_shutdown("Parked!")
            else:
                rospy.logerr(f"Found unknown state: {self.state.name}")
                rospy.signal_shutdown("Unknown State")
            rate.sleep()

    def wait(self):
        rate = rospy.Rate(self.params["parking_waiting_rate"])
        while not rospy.is_shutdown() and self.state == ParkingState.Waiting:
            rate.sleep()
        self.run()

    def shutdown_hook(self):
        print("Shutting down")
        self.twist.v = 0
        self.twist.omega = 0
        for _ in range(9):
            self.vel_pub.publish(self.twist)

    def parking_pid(self, error, P_, D_):
        P = error * P_
        d_error = error - self.parking_last_error / (
            rospy.get_time() - self.parking_last_time
        )
        self.parking_last_error = error
        self.parking_last_time = rospy.get_time()
        D = d_error * D_
        v = P[0] + D[0]
        v = np.clip(
            v, -self.params["parking_max_v"], self.params["parking_max_v"]
        )
        omega = P[1] + D[1]
        omega = np.clip(
            omega,
            -self.params["parking_max_omega"],
            self.params["parking_max_omega"],
        )
        self.twist.v = v
        self.twist.omega = omega

    def parking_stop_state(self):
        self.state_start_time = time.time()
        rate = rospy.Rate(self.params["parking_rate"])
        while (
            time.time() - self.state_start_time
            < self.params["parking_stop_time"]
        ):
            self.twist.v = 0
            self.twist.omega = 0
            self.vel_pub.publish(self.twist)
            rate.sleep()

        self.state = ParkingState.Forward

    def parking_forward_state(self):
        # move forward for a parameterized amount of time
        rate = rospy.Rate(1 / self.params["parking_forward_time"])
        self.twist.v = self.params["parking_forward_constant_v"]
        self.twist.omega = self.params["parking_forward_constant_omega"]
        self.vel_pub.publish(self.twist)
        rate.sleep()

        self.parking_find_perpendicular(self.params["parking_stop_tof_min"])

        # move forward with PID until the tof sensor is at the right distance
        rate = rospy.Rate(self.params["parking_rate"])
        P_ = np.array([self.params["parking_P_x"], 0])
        D_ = np.array([self.params["parking_D_x"], 0])
        depth = self.parking_stall["depth"]
        while True:
            if self.tof_distance >= self.params["max_tof_distance"]:
                self.parking_find_perpendicular()
                continue
            distance_error = (
                self.tof_distance
                - self.params["parking_forward_distance"][depth]
            )
            error = np.array([distance_error, 0])
            if abs(distance_error) < self.params["parking_forward_epsilon"]:
                break
            self.parking_pid(error, P_, D_)
            self.vel_pub.publish(self.twist)
            rate.sleep()

        self.state = ParkingState.Turn

    def parking_find_perpendicular(self, ignore_tof_min=0):
        rate_rotate = rospy.Rate(
            1 / self.params["parking_find_perpendicular_rotate_time"]
        )
        rate_stop = rospy.Rate(
            1 / self.params["parking_find_perpendicular_stop_time"]
        )
        min_tof_distance = self.params["max_tof_distance"]
        step = 0
        max_step = 1
        add = 1
        total_steps = 0
        while True:
            rospy.loginfo(
                f"step / max / total: {step} / {max_step} / {total_steps}"
            )
            rospy.loginfo(f"tof_distance: {self.tof_distance}")
            rospy.loginfo(f"min_tof_distance: {min_tof_distance}")
            self.twist.v = 0
            omega = self.params["parking_find_perpendicular_omega"][str(add)]
            rospy.loginfo(f"omega: {omega}")
            self.twist.omega = omega
            self.vel_pub.publish(self.twist)
            rate_rotate.sleep()
            self.twist.v = 0
            self.twist.omega = 0
            self.vel_pub.publish(self.twist)
            rate_stop.sleep()
            if step == max_step:
                max_step += add
                max_step *= -1
                add *= -1
            step += add
            if (
                self.tof_distance < min_tof_distance
                and self.tof_distance > ignore_tof_min
            ):
                min_tof_distance = self.tof_distance
                step = 0
                max_step = 1
                add = 1
            elif total_steps > 3 and step == 0:
                break
            elif (
                total_steps
                > self.params["parking_find_perpendicular_max_total_steps"]
            ):
                total_steps = 0
                min_tof_distance = self.params["max_tof_distance"]
                step = 0
            total_steps += 1

    def parking_turn_state(self):
        rate = rospy.Rate(self.params["parking_rate"])
        side = self.parking_stall["side"]
        self.twist.v = 0
        self.twist.omega = 0
        rate.sleep()

        # rotate until tof sensor is at max_tof_distance
        while self.tof_distance < self.params["max_tof_distance"]:
            self.twist.omega = self.params["parking_turn_omega"][side]
            self.vel_pub.publish(self.twist)
            rate.sleep()

        # rotate until tof sensor is less than parking_turn_max_tof_distance
        while self.tof_distance > self.params["parking_turn_max_tof_distance"]:
            self.twist.omega = self.params["parking_turn_omega"][side]
            self.vel_pub.publish(self.twist)
            rate.sleep()

        self.parking_find_perpendicular()

        self.state = ParkingState.Reverse

    def parking_reverse_state(self):
        rate = rospy.Rate(self.params["parking_rate"])
        P_ = np.array([self.params["parking_P_x"], 0])
        D_ = np.array([self.params["parking_D_x"], 0])
        while True:
            distance_error = (
                self.tof_distance
                - self.params["parking_reverse_target_tof_distance"]
            )
            error = np.array([distance_error, 0])
            rospy.loginfo(f"DEBUG error: {error}")
            if np.linalg.norm(error) < self.params["parking_reverse_epsilon"]:
                break
            self.parking_pid(error, P_, D_)
            rospy.loginfo(f"DEBUG v: {self.twist.v}")
            rospy.loginfo(f"DEBUG omega: {self.twist.omega}")
            self.vel_pub.publish(self.twist)
            rate.sleep()
        rate = rospy.Rate(1 / self.params["parking_reverse_constant_time"])

        # reverse backwards for a parameterized amount time
        self.twist.v = self.params["parking_reverse_v"]
        self.vel_pub.publish(self.twist)
        rate.sleep()
        self.twist.v = 0
        self.vel_pub.publish(self.twist)
        self.state = ParkingState.ShutDown


if __name__ == "__main__":
    node = ParkingNode("parking_node")
    node.wait()
    rospy.spin()
