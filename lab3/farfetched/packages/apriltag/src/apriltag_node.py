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
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

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
        self.raw_image = None

        # Standard subscribers and publishers
        self.img_pub = rospy.Publisher(
            '~compressed', CompressedImage, queue_size=1
        )

        self.led_pattern_pub = rospy.Publisher(
            f'/{self.hostname}/led_emitter_node/led_pattern',
            LEDPattern,
            queue_size=1,
        )

        self.pub_teleport = rospy.Publisher(
            f"/{self.hostname}/deadreckoning/teleport",
            Transform,
            queue_size=1
        )

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
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

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
            estimate_tag_pose=True,
            camera_params=[
                self.camera_model.fx(),
                self.camera_model.fy(),
                self.camera_model.cx(),
                self.camera_model.cy()
            ],
            tag_size=0.05
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
        for i in range(5):
            rgba = ColorRGBA()
            if 3 <= i <= 4:
                rgba.r = colors[0] / 255
                rgba.g = colors[1] / 255
                rgba.b = colors[2] / 255
                rgba.a = 1.0
            else:
                rgba.r = rgba.g = rgba.b = rgba.a = 0.0

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
        min_distance = np.inf
        closest_tag_id = None
        for detection in detections:
            T_AC = np.eye(4)
            T_AC[:3, :3] = detection.pose_R
            T_AC[:3, 3] = detection.pose_t.flatten()
            translation = tr.translation_from_matrix(T_AC)
            distance = np.linalg.norm(translation, 2)
            q = tr.quaternion_from_matrix(T_AC)
            transform = Transform(
                translation=Vector3(*translation),
                rotation=Quaternion(*q)
            )
            tag_id = detection.tag_id
            transform_stamped = TransformStamped(
                header=Header(
                    stamp=rospy.Time.now(),
                    frame_id=f"{self.hostname}/camera_optical_frame"
                ),
                child_frame_id=f"at_{tag_id}",
                transform=transform,
            )
            if distance < min_distance:
                min_distance = distance
                closest_tag_id = tag_id
            transforms.append(transform_stamped)
        self.tf_broadcaster.sendTransform(transforms)

        if closest_tag_id is not None:
            transform_apriltag_footprint = self.tf_buffer.lookup_transform(
                f"at_{closest_tag_id}",
                f"{self.hostname}/footprint",
                0
            )
            transform_apriltag_static_world = self.tf_buffer.lookup_transform(
                f"at_{closest_tag_id}_static",
                f"{self.hostname}/world",
                0
            )
            transform = transform_apriltag_footprint * transform_apriltag_static_world
            self.pub_teleport.publish(transform)

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
            self.img_pub.publish(message)
            led_message = self.create_led_message(color)
            self.led_pattern_pub.publish(led_message)
            rate.sleep()

    def onShutdown(self):
        super(AprilTagNode, self).onShutdown()


if __name__ == '__main__':
    camera_node = AprilTagNode(node_name='apriltag_node')
    camera_node.run()
