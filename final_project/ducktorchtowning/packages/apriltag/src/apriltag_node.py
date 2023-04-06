#!/usr/bin/env python3
from typing import List

import cv2
import numpy as np
import rospy
import json
from cv_bridge import CvBridge
from dt_apriltags import Detection, Detector
from duckietown.dtros import DTROS, NodeType
from geometry_msgs.msg import Quaternion, Transform, TransformStamped, Vector3
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import ColorRGBA, Header, Float64
from tag import TAG_ID_TO_TAG, Tag, TagType
from tf import transformations as tr
from tf2_ros import (
    Buffer,
    ConnectivityException,
    TransformBroadcaster,
    TransformListener,
)

"""

This is a template that can be used as a starting point for the CRA1 exercise.
You need to project the model file in the 'models' directory on an AprilTag.
To help you with that, we have provided you with the Renderer class that render the obj file.

"""


class AprilTagNode(DTROS):
    def __init__(self, node_name):
        super(AprilTagNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC
        )

        self.hostname = rospy.get_param("~veh")
        self.bridge = CvBridge()

        with open("/params.json") as f:
            self.params = json.load(f)

        self.params = {
            **self.params["default"],
            **(self.params.get(self.hostname) or {}),
        }

        self.detector = Detector(
            searchpath=["apriltags"],
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        self.camera_model = PinholeCameraModel()
        self.raw_image = None

        # Standard subscribers and publishers
        self.img_pub = rospy.Publisher(
            "~compressed", CompressedImage, queue_size=1
        )

        self.pub_teleport = rospy.Publisher(
            f"/{self.hostname}/deadreckoning_node/teleport",
            Transform,
            queue_size=1,
        )

        self.pub_ap_distance = rospy.Publisher(
            f"/{self.hostname}/deadreckoning_node/ap_distance",
            Float64,
            queue_size=1,
        )

        self.pub_ap_position = rospy.Publisher(
            f"/{self.hostname}/ap_node/ap_position",
            Vector3,
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

        self.tf_broadcaster = TransformBroadcaster()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

    def process_image(self, raw):
        """Undistorts raw images."""
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
                self.camera_model.cy(),
            ],
            tag_size=0.05,
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
                2,
            )
            text = str(detection.tag_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            center = detection.center.astype(np.int64)
            font_scale = 1
            font_thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[
                0
            ]
            center[0] -= text_size[0] // 2
            center[1] += text_size[1] // 2
            cv2.putText(
                image, text, center, font, font_scale, bgr, font_thickness
            )

    def rectify(self, image):
        """Undistorts raw images."""
        rectified = np.zeros_like(image)
        self.camera_model.rectifyImage(image, rectified)
        return rectified

    def broadcast_transforms(self, detections: List[Detection]):
        transforms = []
        min_distance = np.inf
        closest_tag_id = None
        for detection in detections:
            T_at_camera = np.eye(4)
            T_at_camera[:3, :3] = detection.pose_R
            T_at_camera[:3, 3] = detection.pose_t.flatten()
            translation = tr.translation_from_matrix(T_at_camera)
            distance = np.linalg.norm(translation, 2)

            q = tr.quaternion_from_matrix(T_at_camera)
            transform = Transform(
                translation=Vector3(*translation), rotation=Quaternion(*q)
            )
            tag_id = detection.tag_id
            transform_stamped = TransformStamped(
                header=Header(
                    stamp=rospy.Time.now(),
                    frame_id=f"{self.hostname}/camera_optical_frame",
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
            try:
                transform_odometry_at = self.tf_buffer.lookup_transform(
                    f"at_{closest_tag_id}",
                    "odometry",
                    rospy.Time(0),
                    rospy.Duration(1.0),
                ).transform
            except ConnectivityException as e:
                rospy.logwarn_throttle(1.0, str(e))
                return
            except:
                rospy.logwarn_throttle(1.0, "Another exception fired")
                return
            T_odometry_at = tr.compose_matrix(
                translate=(
                    [
                        transform_odometry_at.translation.x,
                        transform_odometry_at.translation.y,
                        transform_odometry_at.translation.z,
                    ]
                ),
                angles=tr.euler_from_quaternion(
                    [
                        transform_odometry_at.rotation.x,
                        transform_odometry_at.rotation.y,
                        transform_odometry_at.rotation.z,
                        transform_odometry_at.rotation.w,
                    ]
                ),
            )
            try:
                transform_at_static_world = self.tf_buffer.lookup_transform(
                    "world",
                    f"at_{closest_tag_id}_static",
                    rospy.Time(0),
                    rospy.Duration(1.0),
                ).transform
            except Exception as e:
                rospy.logwarn_throttle(1.0, str(e))
                return
            T_at_static_world = tr.compose_matrix(
                translate=(
                    [
                        transform_at_static_world.translation.x,
                        transform_at_static_world.translation.y,
                        transform_at_static_world.translation.z,
                    ]
                ),
                angles=tr.euler_from_quaternion(
                    [
                        transform_at_static_world.rotation.x,
                        transform_at_static_world.rotation.y,
                        transform_at_static_world.rotation.z,
                        transform_at_static_world.rotation.w,
                    ]
                ),
            )
            T_odometry_world = T_at_static_world @ T_odometry_at
            translation = tr.translation_from_matrix(T_odometry_world)
            q = tr.quaternion_from_matrix(T_odometry_world)
            transform_odometry_world = Transform(
                translation=Vector3(*translation), rotation=Quaternion(*q)
            )

            self.pub_ap_distance.publish(min_distance)
            self.pub_teleport.publish(transform_odometry_world)
            self.pub_ap_position.publish(Vector3(*translation))

    def run(self):
        rate = rospy.Rate(self.params["ap_rate"])

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
            message = self.bridge.cv2_to_compressed_imgmsg(
                image, dst_format="jpeg"
            )
            self.img_pub.publish(message)
            rate.sleep()

    def onShutdown(self):
        super(AprilTagNode, self).onShutdown()


if __name__ == "__main__":
    camera_node = AprilTagNode(node_name="apriltag_node")
    camera_node.run()
