#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from mallard_eye.srv import MallardEyedentify, MallardEyedentifyResponse
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import Vector3
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

        rospy.loginfo("Started mallard eye!")

    def cb_compressed(self, compressed):
        self.compressed = compressed

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
