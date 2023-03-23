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


class MallardEyeNode(DTROS):
    def __init__(self, node_name):
        super(MallardEyeNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC
        )

        self.hostname = rospy.get_param("~veh")
        self.bridge = CvBridge()
        self.net = Net(weights_path="/weights.npy")

        self.serv = rospy.Service(
            "mallard_eyedentification", MallardEyedentify, self.identify
        )
        self.pub = rospy.Publisher(
            f"/{self.hostname}/mallard_eye_node/image/compressed",
            CompressedImage,
            queue_size=1,
        )
        rospy.loginfo("Started led_control_service")

    def cb_compressed(self, compressed):
        self.compressed = compressed

    def identify(self, _: MallardEyedentify) -> MallardEyedentifyResponse:
        raw_image = self.bridge.compressed_imgmsg_to_cv2(self.compressed)
        rectified_image = np.zeros_like(raw_image)
        image = self.camera_model.rectifyImage(raw_image, rectified_image)

        image_warped, corners = warp_image(image)
        if image_warped is None:
            return -1

        image_copy = image.copy()
        image_copy[:64, :64] = cv.resize(image_warped, (64, 64))
        image_copy.drawContours(
            [corners], -1, (0, 255, 255), 2
        )

        image_warped = preprocess_image(image_warped)
        x = normalize_image(image_warped)

        digit = self.net.predict(x)

        cv.putText(
            image_copy,
            str(digit),
            (16, 128),
            cv.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 255),
            2,
        )
        msg = self.bridge.cv2_to_compressed_imgmsg(image_copy)
        self.pub.publish(msg)

        return digit

    def onShutdown(self):
        super(MallardEyeNode, self).onShutdown()


if __name__ == "__main__":
    camera_node = MallardEyeNode(node_name="mallard_eye_node")
    camera_node.run(1)
