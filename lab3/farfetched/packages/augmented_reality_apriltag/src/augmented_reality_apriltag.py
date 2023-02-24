#!/usr/bin/env python3
import numpy as np
import os
import math
import cv2
from renderClass import Renderer

import rospy
import yaml
import sys
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from dt_apriltags import Detector
from image_geometry import PinholeCameraModel

import rospkg


"""

This is a template that can be used as a starting point for the CRA1 exercise.
You need to project the model file in the 'models' directory on an AprilTag.
To help you with that, we have provided you with the Renderer class that render the obj file.

"""

class ARNode(DTROS):
    def __init__(self, node_name):
        super(ARNode, self).__init__(node_name=node_name,
                                     node_type=NodeType.GENERIC)

        self.hostname = rospy.get_param("~veh")
        self.bridge = CvBridge()

        # Initialize an instance of Renderer giving the model in input.
        rospack = rospkg.RosPack()
        self.renderer = Renderer(rospack.get_path('augmented_reality_apriltag')
                                 + '/src/models/duckie.obj')

        at_detector = Detector(searchpath=['apriltags'],
                                    families='tag36h11',
                                    nthreads=1,
                                    quad_decimate=1.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)

        self.detect = lambda img: at_detector.detect(img,
                                                     estimate_tag_pose=False,
                                                     camera_params=None,
                                                     tag_size=None)

        self.camera_model = PinholeCameraModel()

        # Standard subscribers and publishers
        self.pub = rospy.Publisher('~compressed', CompressedImage, queue_size=2)
        self.raw_image = None

        rospy.Subscriber(
            f'/{self.hostname}/camera_node/image/compressed',
            CompressedImage,
            self.april_cb,
        )


        self.camera_info_sub = rospy.Subscriber(
            f"/{self.hostname}/camera_node/camera_info",
            CameraInfo,
            self.callback_camera_info
        )

    def callback_camera_info(self, message):
        """Callback for the camera_node/camera_info topic."""
        self.camera_model.fromCameraInfo(message)

    def april_cb(self, compressed):
        self.raw_image = self.readImage(compressed)


    def pub_loop(self, rate=30):
        rate = rospy.Rate(rate)

        while not rospy.is_shutdown():
            if self.raw_image is None:
                rate.sleep()
                continue
            image = self.raw_image.copy()
            grayscale = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
            tags = self.detect(grayscale)
            for tag in tags:
                projection_matrix = self.projection_matrix(tag.homography)
                image = self.renderer.render(image, projection_matrix)
            message = self.bridge.cv2_to_compressed_imgmsg(
                image, dst_format="jpeg"
            )
            self.pub.publish(message)
            rate.sleep()

    def projection_matrix(self, homography):
        """
            Write here the compuatation for the projection matrix, namely the matrix
            that maps the camera reference frame to the AprilTag reference frame.

            Source:
            https://bitesofcode.wordpress.com/2018/09/16/
            augmented-reality-with-python-and-opencv-part-2/
        """
        camera_parameters = self.camera_model.intrinsicMatrix()
        # Compute rotation along the x and y axis as well as the translation
        rot_and_transl = np.linalg.inv(camera_parameters) @ homography
        col_1 = rot_and_transl[:, 0].T
        col_2 = rot_and_transl[:, 1].T
        col_3 = rot_and_transl[:, 2].T
        # normalise vectors
        l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
        rot_1 = col_1 / l
        rot_2 = col_2 / l
        translation = col_3 / l
        # compute the orthonormal basis
        c = rot_1 + rot_2
        p = np.cross(rot_1, rot_2)
        d = np.cross(c, p)
        rot_1 = c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2) * 1 / math.sqrt(2)
        rot_2 = c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2) * 1 / math.sqrt(2)
        rot_3 = np.cross(rot_1, rot_2)
        # finally, compute the 3D projection matrix from the model to the current frame
        projection = np.stack((rot_1, rot_2, rot_3, translation)).T
        return camera_parameters @ projection

    def readImage(self, msg_image):
        """
            Convert images to OpenCV images
            Args:
                msg_image (:obj:`CompressedImage`) the image from the camera node
            Returns:
                OpenCV image
        """
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg_image)
            return cv_image
        except CvBridgeError as e:
            self.log(e)
            return []

    def readYamlFile(self,fname):
        """
            Reads the 'fname' yaml file and returns a dictionary with its input.

            You will find the calibration files you need in:
            `/data/config/calibrations/`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return


    def onShutdown(self):
        super(ARNode, self).onShutdown()


if __name__ == '__main__':
    camera_node = ARNode(node_name='augmented_reality_apriltag_node')
    camera_node.pub_loop()
