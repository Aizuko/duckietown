#!/usr/bin/env python3
from typing import List

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from dt_apriltags import Detection, Detector
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import LEDPattern
from geometry_msgs.msg import Quaternion, Transform, TransformStamped, Vector3
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import ColorRGBA, Header
from tag import TAG_ID_TO_TAG, Tag, TagType
from tf import transformations as tr
from tf2_ros import TransformBroadcaster

"""

This is a template that can be used as a starting point for the CRA1 exercise.
You need to project the model file in the 'models' directory on an AprilTag.
To help you with that, we have provided you with the Renderer class that render the obj file.

"""


class AprilTagNode(DTROS):
    def __init__(self, node_name):
        super(AprilTagNode, self).__init__(node_name=node_name,
                                           node_type=NodeType.GENERIC)

        self.hostname = rospy.get_param("~veh")
        self.bridge = CvBridge()

        self.detector = Detector(
            searchpath=['apriltags'],
            families='tag36h11',
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

        self.camera_model = PinholeCameraModel()

        # Standard subscribers and publishers
        self.pub = rospy.Publisher(
            '~compressed', CompressedImage, queue_size=1
        )

        self.led_pattern_pub = rospy.Publisher(
            f'/{self.hostname}/led_emitter_node/led_pattern',
            LEDPattern,
            queue_size=10,
        )
        self.raw_image = None

        self.compressed_sub = rospy.Subscriber(
            f'/{self.hostname}/camera_node/image/compressed',
            CompressedImage,
            self.cb_compressed,
        )

        self.camera_info_sub = rospy.Subscriber(
            f"/{self.hostname}/camera_node/camera_info",
            CameraInfo,
            self.cb_camera_info
        )

        self.tf_broadcaster = TransformBroadcaster()

    def process_image(self, raw):
        """Undistorts raw images.
        """
        rectified = np.zeros_like(raw)
        self.camera_model.rectifyImage(raw, rectified)
        return rectified

    def cb_camera_info(self, message):
        """Callback for the camera_node/camera_info topic."""
        self.camera_model.fromCameraInfo(message)

    def cb_compressed(self, compressed):
        self.raw_image = self.bridge.compressed_imgmsg_to_cv2(compressed)

    def detect(self, image):
        return self.detector.detect(
            image,
            estimate_tag_pose=False,
            camera_params=None,
            tag_size=None
        )

    def render_tag(self, image: np.ndarray, detection: Detection):
        for i in range(4):
            tag = TAG_ID_TO_TAG.get(
                detection.tag_id, Tag(detection.tag_id, None)
            )
            corner_a = detection.corners[i]
            corner_b = detection.corners[(i + 1) % 4]
            bgr = tag.color[::-1]
            cv2.line(
                image,
                corner_a.astype(np.int64),
                corner_b.astype(np.int64),
                bgr,
                2
            )
            text = str(detection.tag_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            center = detection.center.astype(np.int64)
            font_scale = 1
            font_thickness = 2
            text_size = cv2.getTextSize(
                text, font, font_scale, font_thickness)[0]
            center[0] -= text_size[0] // 2
            center[1] += text_size[1] // 2
            cv2.putText(
                image,
                text,
                center,
                font,
                font_scale,
                bgr,
                font_thickness
            )

    def get_led_color(self, detections: List[Detection]):
        white = (255, 255, 255)
        priority_tag = None
        for detection in detections:
            tag = TAG_ID_TO_TAG.get(
                detection.tag_id, Tag(
                    detection.tag_id, None))
            if tag is None:
                rospy.logwarn(f"Unknown tag id: {tag.tag_id}")
                continue
            if tag.type is TagType.StopSign:
                priority_tag = tag
                break
            if tag.type is TagType.TIntersection:
                priority_tag = tag
            elif tag.type is TagType.UofA and priority_tag is None:
                priority_tag = tag

        return priority_tag.color if priority_tag else white

    def create_led_message(self, colors: 'List[float]') -> 'LEDPattern':
        """ Creates an led message with the colors set to values from a tuple

        Args:
            colors (list[float]): RGB values from 0 to 255
        """
        led_message = LEDPattern()
        for _ in range(5):
            rgba = ColorRGBA()
            rgba.r = colors[0] / 255
            rgba.g = colors[1] / 255
            rgba.b = colors[2] / 255
            rgba.a = 1.0
            led_message.rgb_vals.append(rgba)
        return led_message

    def rectify(self, image):
        """Undistorts raw images.
        """
        rectified = np.zeros_like(image)
        self.camera_model.rectifyImage(image, rectified)
        return rectified

    def broadcast_transforms(self, detections: List[Detection]):
        transforms = []
        for detection in detections:
            H = detection.homography
            translation = tr.translation_from_matrix(H)
            q = tr.quaternion_from_matrix(H)
            transform = TransformStamped(
                header=Header(
                    stamp=rospy.Time.now(),
                    frame_id=f"{self.hostname}/camera_optical_frame"
                ),
                child_frame_id=f"at_{detection.tag_id}",
                transform=Transform(
                    translation=Vector3(*translation),
                    rotation=Quaternion(*q)
                ),
            )
            transforms.append(transform)
        self.tf_broadcaster.sendTransform(transforms)

    def run(self, rate=30):
        rate = rospy.Rate(rate)

        while not rospy.is_shutdown():
            if self.raw_image is None:
                rate.sleep()
                continue
            image = self.raw_image.copy()
            image = self.rectify(image)
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = self.detect(grayscale)
            for detection in detections:
                self.render_tag(image, detection)
            self.broadcast_transforms(detections)
            color = self.get_led_color(detections)
            message = self.bridge.cv2_to_compressed_imgmsg(
                image, dst_format="jpeg"
            )
            self.pub.publish(message)
            led_message = self.create_led_message(color)
            self.led_pattern_pub.publish(led_message)
            rate.sleep()

    def onShutdown(self):
        super(AprilTagNode, self).onShutdown()


if __name__ == '__main__':
    camera_node = AprilTagNode(node_name='apriltag_node')
    camera_node.run()
