#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from mallard_eye.srv import MallardEyedentify, MallardEyedentifyResponse
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import Vector3
from nn import Net
from preprocess import normalize_image, preprocess_image, warp_image
from image_geometry import PinholeCameraModel


class MallardEyeNode(DTROS):
    def __init__(self, node_name):
        super(MallardEyeNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC
        )

        self.hostname = rospy.get_param("~veh")
        self.bridge = CvBridge()
        self.camera_model = PinholeCameraModel()
        self.net = Net(weights_path="/weights.npy")
        self.annotated_image = None
        self.compressed = None
        self.is_camera_init = False
        self.ap_position = None

        self.pub = rospy.Publisher(
            f"/{self.hostname}/mallard_eye_node/image/compressed",
            CompressedImage,
            queue_size=1,
        )

        self.serv = rospy.Service(
            "mallard_eyedentification", MallardEyedentify, self.identify
        )

        self.compressed_sub = rospy.Subscriber(
            f"/{self.hostname}/camera_node/image/compressed",
            CompressedImage,
            self.cb_compressed,
            queue_size=1,
        )

        self.camera_info_sub = rospy.Subscriber(
            f"/{self.hostname}/camera_node/camera_info",
            CameraInfo,
            self.cb_camera_info,
            queue_size=1,
        )

        self.ap_position_sub = rospy.Subscriber(
            f"/{self.hostname}/ap_node/ap_position",
            Vector3,
            self.cb_ap_position,
            queue_size=1,
        )

        rospy.loginfo("Started mallard eye!")

    def cb_compressed(self, compressed):
        self.compressed = compressed

    def cb_ap_position(self, vec):
        self.ap_position = [vec.x, vec.y, vec.z]

    def cb_camera_info(self, message):
        """Callback for the camera_node/camera_info topic."""
        self.camera_model.fromCameraInfo(message)
        self.camera_info_sub.unregister()
        self.is_camera_init = True

    def set_compressed(
        self,
        image: np.ndarray,
        image_warped: np.ndarray,
        corners: np.ndarray,
        digit: int,
    ):
        image_copy = image.copy()
        image_copy[:64, :64] = np.repeat(
            cv.resize(image_warped, (64, 64))[:, :, np.newaxis], 3, axis=2
        )
        cv.drawContours(image_copy, [np.int64(corners)], -1, (0, 255, 255), 2)

        cv.putText(
            image_copy,
            str(digit),
            (16, 128),
            cv.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 255),
            2,
        )
        self.annotated_image = self.bridge.cv2_to_compressed_imgmsg(image_copy)

    def identify(self, _) -> MallardEyedentifyResponse:
        if not self.is_camera_init:
            return -3
        if self.compressed is not None:
            raw_image = self.bridge.compressed_imgmsg_to_cv2(self.compressed)
            rectified_image = np.zeros_like(raw_image)
            self.camera_model.rectifyImage(raw_image, rectified_image)

            image_warped, corners = warp_image(rectified_image)
            if image_warped is None:
                return -1

            image_warped = preprocess_image(image_warped)
            x = normalize_image(image_warped)

            digit = self.net.predict(x)
            self.set_compressed(rectified_image, image_warped, corners, digit)

            print(f"Apriltag position in world frame {self.ap_position}")
            return digit
        else:
            rospy.loginfo("Starting a detection -2")
            return -2  # Error code for not having an image

    def run(self, rate=1):
        rate = rospy.Rate(rate)

        while not rospy.is_shutdown():
            if self.annotated_image is not None:
                self.pub.publish(self.annotated_image)
            rate.sleep()

    def onShutdown(self):
        super(MallardEyeNode, self).onShutdown()


if __name__ == "__main__":
    camera_node = MallardEyeNode(node_name="mallard_eye_node")
    camera_node.run(1)
