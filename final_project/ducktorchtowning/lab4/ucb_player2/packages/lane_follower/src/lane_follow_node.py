#!/usr/bin/env python3
import json
import time
from collections import deque
from enum import IntEnum, Enum, auto, unique
from functools import lru_cache

import cv2
import cv2 as cv
import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped, WheelsCmdStamped
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, CompressedImage, Range
from std_msgs.msg import ColorRGBA
from std_msgs.msg import Float32, Float64
from lane_follower.srv import StartParking, StartParkingResponse
from cv_bridge import CvBridge
from geometry_msgs.msg import Transform, Vector3, TransformStamped
from tf2_ros import Buffer, TransformListener
from tf import transformations as tr


# ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
ROAD_MASK = [(10, 60, 165), (40, 255, 255)]
# STOP_MASK = [(0, 70, 150), (20, 255, 255)]
STOP_MASK = [(0, 100, 120), (10, 255, 255)]
# DUCKIES_ONLY = [(0, 55, 145), (20, 255, 255)]
DUCKIES_ONLY = [(10, 55, 145), (20, 255, 255)]
BLUELINE = [(85, 60, 85), (120, 255, 160)]

OFF_COLOR = ColorRGBA()
OFF_COLOR.r = OFF_COLOR.g = OFF_COLOR.b = OFF_COLOR.a = 0.0


with open("/params.json") as f:
    params = json.load(f)["default"]


@unique
class DS(IntEnum):
    """States our duckiebot can visit. These modify the LaneFollowNode"""

    WaitForInitialAP = 1
    LaneFollowing = 2
    Stopped = 3
    BlindTurnLeft = 4
    BlindTurnRight = 5
    BlindForward = 6
    Tracking = 7
    LondonStyle = 8
    WaitForCrossing = 9

    ShuttingDown = 90
    ExitForParking = 100


class TagType(IntEnum):
    """IntEnum mirror of the classification @ apriltag tag.py"""

    NotImportant = 0
    RightStop = 1
    LeftStop = 2
    ForwardStop = 3
    CrossingStop = 4
    ParkingLotEnteringStop = 5


with open("/params.json") as f:
    params = json.load(f)["default"]


class SeenAP:
    """Tuple for an apriltag detection"""

    def __init__(self, tag: TagType, distance: float):
        self.tag = tag
        self.distance = distance
        self.time = time.time()

    def is_within_time(self) -> bool:
        return time.time() - self.time < params["ap_stale_timeout"]

    def is_within_distance(self) -> bool:
        return self.distance < params["ap_considered_distance"]

    def is_within_criteria(self) -> bool:
        return self.is_within_time() and self.is_within_distance()


class SeenAP:
    """Tuple for an apriltag detection"""

    def __init__(self, tag: TagType, distance: float):
        self.tag = tag
        self.distance = distance
        self.time = time.time()

    def is_within_time(self) -> bool:
        return time.time() - self.time < params["ap_stale_timeout"]

    def is_within_distance(self) -> bool:
        return self.distance < params["ap_considered_distance"]

    def is_within_criteria(self) -> bool:
        return self.is_within_time() and self.is_within_distance()


class FrozenClass(object):
    __isfrozen = False

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError(f"{self} is a frozen class")
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True


class LaneFollowNode(DTROS, FrozenClass):
    def __init__(self, node_name):
        super(LaneFollowNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC
        )

        # ╔────────────────────────────────────────────────────────────────────╗
        # │  Cδηsταητs (τδ τμηε)                                               |
        # ╚────────────────────────────────────────────────────────────────────╝
        # Utils
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")
        self.bridge = CvBridge()

        self.params = params

        self.is_american = True
        self.is_debug = self.params["is_debug"]
        self.last_seen_ap = None

        # Lane following
        self.offset = self.params["lane_follow_offset"]

        self.min_velocity = self.params["min_velocity"]
        self.velocity = self.params["velocity"]
        self.max_velocity = self.params["max_velocity"]
        self.twist = Twist2DStamped(v=self.velocity, omega=0)
        self.Px = self.params["Px"]
        self.Dx = self.params["Dx"]

        # New bits!!!
        self.london_start_time = None
        self.has_seen_parking = False
        self.saw_first_ap_time = None
        self.tracking_start = None
        self.is_started_helping = False
        self.finished_helping = False
        self.finish_help_time = None
        self.london_stage = 0

        # Stopping
        self.stop_duration = self.params["stop_duration"]
        self.stop_immunity = self.params["stop_immunity"]
        self.tracking_distance = self.params["tracking_distance"]
        self.tracking_timeout = self.params["tracking_timeout"]
        self.stopline_area_min = self.params["stopline_area_min"]
        self.stopline_area_max = self.params["stopline_area_max"]

        # Tracking
        self.safe_distance = self.params["safe_distance"]
        self.blind_duration_forward = self.params["blind_duration_forward"]
        self.blind_duration_left = self.params["blind_duration_left"]
        self.blind_duration_right = self.params["blind_duration_right"]

        # ╔────────────────────────────────────────────────────────────────────╗
        # │ Dyηαmic ναriαblεs                                                  |
        # ╚────────────────────────────────────────────────────────────────────╝
        # State
        self.state = DS(self.params["starting_state"])

        # PID Variables
        self.error = None  # Error off target

        self.last_error = 0
        self.last_time = time.time()

        # Stopline variables
        self.last_stop_time = None
        self.next_blind_state = None
        self.blind_start_time = None
        self.is_stop_line = False
        self.image = None

        # TOF
        self.tof_dist = [0.0, 0.0, 0.0]

        # Transform
        self.robot_transform_queue = deque(maxlen=self.params["deque_maxlen"])
        self.robot_transform_time = None

        self.tracking_error = None
        self.tracking_last_error = 0
        self.tracking_last_time = time.time()

        self.Pz = self.params["Pz"]
        self.Dz = self.params["Dz"]

        # Shutdown hook
        rospy.on_shutdown(self.on_shutdown)

        # ╔────────────────────────────────────────────────────────────────────╗
        # │ Pμblishεrs & Sμbscribεrs                                           |
        # ╚────────────────────────────────────────────────────────────────────╝
        self.pub = rospy.Publisher(
            f"/{self.veh}/output/image/mask/compressed",
            CompressedImage,
            queue_size=1,
        )
        self.pub_red = rospy.Publisher(
            f"/{self.veh}/output/image/red/compressed",
            CompressedImage,
            queue_size=1,
        )
        self.vel_pub = rospy.Publisher(
            f"/{self.veh}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1,
        )
        self.sub = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.ajoin_callback,
            queue_size=1,
            buff_size="20MB",
        )
        self.tof_sub = rospy.Subscriber(
            f"/{self.veh}/front_center_tof_driver_node/range",
            Range,
            self.tof_callback,
            queue_size=1,
        )
        self.robot_ahead_transform_sub = rospy.Subscriber(
            f"/{self.veh}/duckiebot_distance_node/transform",
            TransformStamped,
            self.robot_ahead_transform_callback,
            queue_size=1,
        )
        self.ap_sub = rospy.Subscriber(
            f"/{self.veh}/ap_node/ap_detection",
            Vector3,
            self.ap_callback,
            queue_size=1,
        )
        self.start_parking_serv = rospy.ServiceProxy(
            f"/{self.veh}/parking_node/start", StartParking
        )

        self._freeze()  # Now disallow any new attributes

    def ap_callback(self, msg):
        # print(f"Saw an ap with {msg.y}")
        self.last_seen_ap = SeenAP(TagType(int(msg.y)), msg.x)

        if self.last_seen_ap.tag == TagType.ParkingLotEnteringStop:
            self.has_seen_parking = True

    def ajoin_callback(self, msg):
        self.lane_callback(msg)

        if self.state is not DS.Stopped and self.state != DS.WaitForCrossing:
            self.stop_callback(msg)

    def lane_callback(self, msg):
        img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        self.image = img

        crop = img[300:-1, :, :]
        crop_width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # ╔────────────────────────────────────────────────────────────────────╗
        # │ Sεαrch fδr lαηε iη frδητ                                           |
        # ╚────────────────────────────────────────────────────────────────────╝
        areas = np.array([cv2.contourArea(a) for a in contours])

        if len(areas) == 0 or np.max(areas) < 20:
            self.error = None
        else:
            max_idx = np.argmax(areas)

            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                self.error = cx - int(crop_width / 2) + self.offset
                if self.is_debug:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except Exception:
                pass

        if self.is_debug:
            rect_img_msg = self.bridge.cv2_to_compressed_imgmsg(crop)
            self.pub.publish(rect_img_msg)

        # ╔────────────────────────────────────────────────────────────────────╗
        # │ Sεαrch Dμckiεs crδssiηg                                            |
        # ╚────────────────────────────────────────────────────────────────────╝
        if self.state == DS.Stopped or self.state == DS.WaitForCrossing:
            return

        is_seen_crossing_duck = self.is_seen_crossing()

        is_seen_crossing_ap = (
            self.last_seen_ap is not None
            and self.last_seen_ap.tag == TagType.CrossingStop
        )

        if is_seen_crossing_duck and is_seen_crossing_ap:
            # print("BOTH")
            self.state = DS.WaitForCrossing
        # elif is_seen_crossing_duck:
        #    print("Saw duckies crossing, but not ap tag")
        # elif is_seen_crossing_ap:
        #    print("Saw crosssing ap tag, but no ducks")

    def is_seen_crossing(self):
        image = self.image[300:-100, :, :]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, DUCKIES_ONLY[0], DUCKIES_ONLY[1])
        image = cv2.bitwise_and(image, image, mask=mask)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image[image != 0] = 1

        if self.state == DS.WaitForCrossing:
            print(f"Sum: {np.sum(image)} / {params['crossing_sum_thresh']}")
        return np.sum(image) > params["crossing_sum_thresh"]

    def tof_callback(self, msg):
        self.tof_dist.append(msg.range)  # Keep full backlog

    def robot_ahead_transform_callback(self, msg: TransformStamped):
        rospy.loginfo(f"Got transform callback {msg.header.stamp.to_sec()}")
        transform = msg.transform
        T = tr.compose_matrix(
            translate=(
                [
                    transform.translation.x,
                    transform.translation.y,
                    transform.translation.z,
                ]
            ),
            angles=tr.euler_from_quaternion(
                [
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.rotation.w,
                ]
            ),
        )
        self.robot_transform_queue.append(T)

        self.robot_transform_time = msg.header.stamp.to_sec()

    def stop_callback(self, msg):
        if self.is_stop_immune():
            return
        elif (
            self.last_seen_ap is not None
            and self.last_seen_ap.tag == TagType.CrossingStop
        ):
            pass

        img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        crop = img[400:-1, :, :]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, STOP_MASK[0], STOP_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        areas = np.array([cv2.contourArea(a) for a in contours])
        is_stopline = np.any(
            np.logical_and(
                self.stopline_area_min < areas, areas < self.stopline_area_max
            )
        )

        if is_stopline and self.state is DS.LaneFollowing:
            self.is_stop_line = True
        elif not is_stopline and self.is_stop_line:
            self.state = DS.Stopped
            self.last_stop_time = time.time()
            self.is_stop_line = False

        if self.is_debug:
            rect_img_msg = self.bridge.cv2_to_compressed_imgmsg(crop)
            self.pub_red.publish(rect_img_msg)

    def wait_for_crossing(self):
        print("Looking at crossing")
        cross_rate = rospy.Rate(self.params["crossing_interval"])

        consecutive_clears = 0

        while True:
            is_curr_clear = not self.is_seen_crossing()

            if is_curr_clear:
                consecutive_clears += 1
            else:
                consecutive_clears = 0

            if consecutive_clears >= self.params["crossing_consec_thresh"]:
                print("All clear!!! Onwards")
                break

            cross_rate.sleep()

        start_time = time.time()
        rate2 = rospy.Rate(10)

        self.twist.v = 0.2
        self.twist.omega = 0.0
        while time.time() - start_time < self.params["crossing_forward_time"]:
            self.vel_pub.publish(self.twist)
            rate2.sleep()

        for _ in range(4):
            self.stop_wheels()
        rate2.sleep()
        rate2.sleep()
        rate2.sleep()
        rate2.sleep()

    def is_stop_immune(self):
        if self.state == DS.ExitForParking or self.state == DS.LondonStyle:
            return True
        elif self.last_stop_time is None:
            return False
        else:
            return time.time() - self.last_stop_time < params["stop_immunity"]

    def drive_bindly(self):
        self.last_error = self.error = 0
        self.twist.v = self.velocity

        if self.state is DS.BlindForward:
            self.twist.omega = 0
        elif self.state is DS.BlindTurnLeft:
            self.twist.omega = self.params["rot_omega_l"] * np.pi
        elif self.state is DS.BlindTurnRight:
            self.twist.omega = -self.params["rot_omega_r"] * np.pi
        else:
            raise Exception(f"Invalid state `{self.state}` for blind driving")

        self.vel_pub.publish(self.twist)

    def london_style(self):
        # Plan
        # 1. Go a bit back and turn left
        # 2. Stop
        # 3. Go a bit forward
        # 4. English driver for Ns
        # 5. Stop
        # 6. Turn a bit right
        # 7. Go a bit forward (duplicate 2)
        # 8. Stop (duplicate 5)
        london_night = rospy.Rate(self.params["london_sleep_time"])
        rate2 = rospy.Rate(self.params["london_rate"])

        for _ in range(4):
            self.stop_wheels()
        london_night.sleep()

        # if self.london_stage == 0:
        if True:
            # ==== 1. Go a bit back and turn left ====
            self.twist.omega = self.params["london_rev_o"]
            self.twist.v = self.params["london_rev_v"]

            start_time = time.time()

            while time.time() - start_time < self.params["london_back_time"]:
                self.vel_pub.publish(self.twist)
                rate2.sleep()

            # ==== 2. Stop ====
            for _ in range(4):
                self.stop_wheels()
            london_night.sleep()

            # ==== 3. Go a bit forward ====
            self.twist.omega = self.params["london_forward_o"]
            self.twist.v = self.params["london_forward_v"]

            start_time = time.time()

            while time.time() - start_time < self.params["london_forward_time"]:
                self.vel_pub.publish(self.twist)
                rate2.sleep()

            # ==== 4. English driver for Ns ====
            self.london_stage = 1
            # self.london_start_time = time.time()
            # self.state = DS.LondonStyle
            self.twist.omega = self.params["london_forward3_o"]
            self.twist.v = self.params["london_forward3_v"]

            start_time = time.time()

            while (
                time.time() - start_time < self.params["london_forward3_time"]
            ):
                self.vel_pub.publish(self.twist)
                rate2.sleep()

        if True:
            # ==== 5. Stop (duplicate 2) ====
            for _ in range(4):
                self.stop_wheels()
            london_night.sleep()

            # ==== 6. Go forward and turn a bit right ====
            self.twist.omega = self.params["london_forward2_o"]
            self.twist.v = self.params["london_forward2_v"]

            start_time = time.time()

            while (
                time.time() - start_time < self.params["london_forward2_time"]
            ):
                self.vel_pub.publish(self.twist)
                rate2.sleep()

            # ==== 8. Stop (duplicate 2) ====
            for _ in range(4):
                self.stop_wheels()
            london_night.sleep()

            self.london_stage = 0
            return

    def pid_x(self, p_coef=1):
        if self.error is None:
            self.twist.omega = 0
        else:
            # P Term
            P = -self.error * self.Px

            # D Term
            d_error = (self.error - self.last_error) / (
                time.time() - self.last_time
            )
            self.last_error = self.error
            self.last_time = time.time()
            D = d_error * self.Dx

            self.twist.omega = P + D

    def pid_z(self):
        return
        distance_to_robot_ahead = self.distance_to_robot_ahead()

        if distance_to_robot_ahead is None:
            return
        self.tracking_error = self.safe_distance - distance_to_robot_ahead

        if self.tracking_last_error is None:
            self.tracking_last_error = self.tracking_error

        Pz = -self.tracking_error * self.Pz
        d_error = self.tracking_error - self.tracking_last_error
        d_time = time.time() - self.tracking_last_time
        self.tracking_last_error = self.tracking_error
        self.tracking_last_time = time.time()
        Dz = d_error / d_time * self.Dz
        v = Pz + Dz
        v = np.sign(v) * np.clip(
            np.abs(v), self.min_velocity, self.max_velocity
        )
        self.twist.v = np.max((v, self.params["clip_velocity"]))

    def follow_lane(self):
        self.pid_x()
        self.twist.v = self.velocity

        self.vel_pub.publish(self.twist)

    def stop_wheels(self):
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)

    def track_bot(self):
        self.twist.v = self.twist.omega = 0
        self.vel_pub.publish(self.twist)

    def is_robot_ahead(self):
        """Returns true if a robot is ahead within distance"""
        if self.robot_transform_time is None:
            return False
        elif (
            time.time() - self.robot_transform_time
            > self.params["robot_ahead_stale_time"]
        ):
            return False

        latest_transform = self.robot_transform_queue[-1]
        latest_translate = latest_transform[:3, 3]
        if self.params["print_distance"]:
            rospy.loginfo(
                f"{np.linalg.norm(latest_translate)}, {self.tof_dist[-1]}"
            )

        dist = np.linalg.norm(latest_translate)
        return dist <= self.tracking_distance

    def run(self):
        rate = rospy.Rate(self.params["run_rate"])

        while not rospy.is_shutdown():
            rospy.loginfo_throttle(1, f"==== STATE: {self.state.name} ====")

            # ==== Initial Wait ====
            if self.state == DS.WaitForInitialAP:
                if self.last_seen_ap is None:
                    pass
                elif self.saw_first_ap_time is None:
                    # print("Saw ap tag now")
                    self.saw_first_ap_time = time.time()
                elif (
                    time.time() - self.saw_first_ap_time
                    > params["init_wait_time"]
                ):
                    self.state = DS.LaneFollowing
            # ==== Lane following ====
            elif self.state is DS.LaneFollowing:
                self.follow_lane()

                try:
                    if self.is_robot_ahead() and not self.is_started_helping:
                        self.state = DS.Tracking
                        self.is_started_helping = True
                except TypeError:
                    pass
            # ==== London following ====
            elif self.state is DS.LondonStyle:
                if (
                    time.time() - self.london_start_time
                    > self.params["london_driving_time"]
                ):
                    self.state = DS.Tracking
                    continue
            # ==== Stopping ====
            elif self.state is DS.Stopped:
                if self.has_seen_parking:
                    self.state = DS.ExitForParking
                    for _ in range(9):
                        self.stop_wheels()
                    continue
                elif self.last_seen_ap is None:
                    self.state = DS.LaneFollowing
                    print(
                        "Failed to see an ap tag before this turn... Going to keep following"
                    )
                elif self.last_seen_ap.tag == TagType.CrossingStop:
                    cross_rate2 = rospy.Rate(2)
                    for _ in range(8):
                        cross_rate2.sleep()
                    self.state = DS.LaneFollowing
                elif self.last_seen_ap.tag == TagType.ForwardStop:
                    self.state = DS.BlindForward
                elif self.last_seen_ap.tag == TagType.LeftStop:
                    self.state = DS.BlindTurnLeft
                elif self.last_seen_ap.tag == TagType.RightStop:
                    self.state = DS.BlindTurnRight

                self.last_seen_ap = None
                continue
            elif self.state in (
                DS.BlindTurnLeft,
                DS.BlindTurnRight,
                DS.BlindForward,
            ):
                if self.state is DS.BlindTurnLeft:
                    blind_duration = self.params["blind_duration_left"]
                elif self.state is DS.BlindTurnRight:
                    blind_duration = self.params["blind_duration_right"]
                else:
                    blind_duration = self.params["blind_duration_forward"]

                if self.blind_start_time is None:
                    self.blind_start_time = time.time()
                elif time.time() - self.blind_start_time > blind_duration:
                    self.blind_start_time = None
                    self.state = DS.LaneFollowing
                else:
                    self.drive_bindly()
            elif self.state == DS.Tracking:
                self.stop_wheels()
                self.london_style()
                self.state = DS.LaneFollowing
            elif self.state == DS.WaitForCrossing:
                for _ in range(8):
                    self.stop_wheels()
                self.wait_for_crossing()
                self.state = DS.LaneFollowing
            elif self.state == DS.ExitForParking:
                print("It's all on you steven. Good luck!")
                self.start_parking_serv()
                break
            else:
                print(f"===! {self.state.name} !===")
                if self.last_seen_ap is not None:
                    print(f"Saw {self.last_seen_ap.tag.name}")
                else:
                    print("Didn't see a last ap tag")

                rospy.signal_shutdown("Saw an AP tag")

            rate.sleep()
            continue

        self.sub.unregister()
        self.tof_sub.unregister()
        self.robot_ahead_transform_sub.unregister()
        self.ap_sub.unregister()
        return

        # ==================================================================
        # ==================================================================
        # ==================================================================

        """
        if self.state is DS.LondonStyle:
            print("In london")
            if time.time() - self.finish_help_time > self.params["london_time"]:
                print("FINISHED LONDON")
                self.state = DS.LaneFollowing
                self.offset = 220
                continue
            else:
                self.follow_lane()

        elif self.state is DS.LaneFollowing:
            self.set_leds(LEDColor.Green, LEDIndex.Back)
            self.follow_lane()
            try:
                if (
                    self.distance_to_robot_ahead() is not None
                    and self.distance_to_robot_ahead() <= self.tracking_distance
                    and not self.finished_helping
                ):
                    self.state = DS.Tracking
            except TypeError:
                pass

        elif self.state is DS.Stopped:
            self.set_leds(LEDColor.Red, LEDIndex.Back)
            if not self.is_stop_immune():
                if self.robot_transform_time is None:
                    self.state = DS.LaneFollowing
                elif (
                    time.time() - self.robot_transform_time
                    < self.params["inter_plan_time"]
                ):
                    self.state = self.next_blind_state
                else:
                    self.state = DS.LaneFollowing
            else:
                self.stop_wheels()

        elif self.state in (
            DS.BlindTurnLeft,
            DS.BlindTurnRight,
            DS.BlindForward,
        ):
            if self.state is DS.BlindTurnLeft:
                blind_duration = self.params["blind_duration_left"]
            elif self.state is DS.BlindTurnRight:
                blind_duration = self.params["blind_duration_right"]
            else:
                blind_duration = self.params["blind_duration_forward"]

            if self.blind_start_time is None:
                self.blind_start_time = time.time()
            elif time.time() - self.blind_start_time > blind_duration:
                self.blind_start_time = None
                self.state = DS.LaneFollowing
            else:
                self.drive_bindly()

        elif self.state is DS.Tracking:
            print("Tracking")

            if self.tracking_start is None:
                self.tracking_start = time.time()
            elif (
                time.time() - self.tracking_start > self.params["wait_to_help"]
            ):
                print("Ended waiting to help")
                self.finished_helping = True
                self.finish_help_time = time.time()
                self.set_leds(LEDColor.Magenta, LEDIndex.Back)
                self.state = DS.LondonStyle
                self.offset = -220
                continue

            self.set_leds(LEDColor.Blue, LEDIndex.Back)
            self.track_bot()

        else:
            raise Exception(f"Invalid state {self.state}")
        """

    def on_shutdown(self):
        print("SHUTTING DOWN")
        rospy.signal_shutdown("Taking down lane follower")
        self.twist.v = 0
        self.twist.omega = 0

        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)


if __name__ == "__main__":
    node = LaneFollowNode("lanefollow_node")
    rospy.on_shutdown(node.on_shutdown)
    node.run()
    rospy.spin()  # Just in case?
