#!/usr/bin/env python3
import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from preprocess import warp_image, normalize_image, preprocess_image
from nn import Net

from mallard_eye.srv import MallardEyedentify, MallardEyedentifyResponse


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
        rospy.loginfo("Started led_control_service")

    def cb_compressed(self, compressed):
        self.compressed = compressed

    def identify(self, _: MallardEyedentify) -> MallardEyedentifyResponse:
        raw_image = self.bridge.compressed_imgmsg_to_cv2(self.compressed)
        rectified_image = np.zeros_like(raw_image)
        image = self.camera_model.rectifyImage(raw_image, rectified_image)

        image = warp_image(image)
        if image is None:
            return -1

        image = preprocess_image(image)
        x = normalize_image(image)

        digit = self.net.predict(x)
        return digit

    def onShutdown(self):
        super(MallardEyeNode, self).onShutdown()


if __name__ == "__main__":
    camera_node = MallardEyeNode(node_name="mallard_eye_node")
    camera_node.run(1)
