#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from mallard_eye.srv import MallardEyedentify, MallardEyedentifyResponse
from sensor_msgs.msg import CompressedImage
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

        rospy.loginfo("Started mallard eye")

    def cb_compressed(self, compressed):
        self.compressed = compressed

    def set_compressed(
        self,
        image: np.ndarray,
        image_warped: np.ndarray,
        corners: np.ndarray,
        digit: int,
    ):
        image_copy = image.copy()
        image_copy[:64, :64] = np.repeat(
            cv.resize(image_warped, (32, 32))[:, :, np.newaxis], 3, axis=2
        )
        image_copy.drawContours([corners], -1, (0, 255, 255), 2)

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
        rospy.loginfo(f"Starting a detection 1 with camera type {type(self.camera_model)}")
        if self.compressed is not None:
            rospy.loginfo("Starting a detection 2")
            raw_image = self.bridge.compressed_imgmsg_to_cv2(self.compressed)
            rospy.loginfo("Starting a detection 3")
            rectified_image = np.zeros_like(raw_image)
            rospy.loginfo("Starting a detection 4")
            rospy.loginfo(f"Types: {type(raw_image)}, {type(rectified_image)}")
            self.camera_model.rectifyImage(raw_image, rectified_image)
            rospy.loginfo("Starting a detection 5")

            image_warped, corners = warp_image(rectified_image)
            rospy.loginfo("Starting a detection 6")
            if image_warped is None:
                rospy.loginfo("Starting a detection -1")
                return -1

            rospy.loginfo("Starting a detection 7")
            image_warped = preprocess_image(image_warped)
            rospy.loginfo("Starting a detection 8")
            x = normalize_image(image_warped)
            rospy.loginfo("Starting a detection 9")

            digit = self.net.predict(x)
            rospy.loginfo("Starting a detection 10")
            self.set_compressed(image, image_warped, corners, digit)
            rospy.loginfo("Starting a detection 11")

            rospy.loginfo(f"Prediction: {digit}")
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
