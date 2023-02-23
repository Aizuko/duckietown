import cv2
import numpy as np
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo

class Augmenter:
    def __init__(self) -> None:
        self.defined_colors = {
            'red': ['rgb', [1, 0, 0]],
            'green': ['rgb', [0, 1, 0]],
            'blue': ['rgb', [0, 0, 1]],
            'yellow': ['rgb', [1, 1, 0]],
            'magenta': ['rgb', [1, 0, 1]],
            'cyan': ['rgb', [0, 1, 1]],
            'white': ['rgb', [1, 1, 1]],
            'black': ['rgb', [0, 0, 0]]
        }

        self.camera_model = PinholeCameraModel()

    def from_camera_info(self, camera_info_msg: CameraInfo):
        """Updates camera_model with camera_info
        """
        self.camera_model.fromCameraInfo(camera_info_msg)

    def process_image(self, raw):
        """Undistorts raw images.
        """
        rectified = np.zeros_like(raw)
        self.camera_model.rectifyImage(raw, rectified)
        return rectified

    def point_to_pixel(self, image: np.ndarray, point: list):
        """Converts a map file point into coordiantes

        Args:
            point (list): point from map file

        Returns:
            tuple: coordinates in image
        """
        reference_frame, coordinates = point
        if reference_frame == "axle":
            raise NotImplementedError
        elif reference_frame == "camera":
            raise NotImplementedError
        elif reference_frame == "image01":
            pixel = np.array([
                (image.shape[1] - 1) * coordinates[0],
                (image.shape[0] - 1) * coordinates[1]
            ])
            return pixel
        else:
            raise ValueError("Invalid reference frame")

    def ground2pixel(self, ground_coordinates):
        """
        transforms points in ground coordinates (i.e. the robot reference frame)
        to pixels in the image.
        """
        return None


    def render_segments(self, image: np.ndarray, cvmap: dict):
        """Plots the segments from the map files onto the image.
        """
        points = cvmap["points"]
        segments = cvmap["segments"]

        for segment in segments:
            p1 = self.point_to_pixel(image, points[segment["points"][0]])
            p2 = self.point_to_pixel(image, points[segment["points"][1]])
            colo = self.color_to_bgr(segment["color"])

            image = self.draw_segment(image, p1, p2, colo)

        return image

    def color_to_bgr(self, color: str):
        _, [r, g, b] = self.defined_colors[color]
        return (b * 255, g * 255, r * 255)

    def draw_segment(
            self,
            image: np.ndarray,
            pt_a: np.ndarray,
            pt_b: np.ndarray,
            color: tuple):
        """Draw segment on image

        based off https://docs.duckietown.org/daffy/
        duckietown-classical-robotics/out/
        cra_basic_augmented_reality_exercise.html

        Args:
            image (np.ndarray): image to draw on
            pt_a (np.ndarray): start point of segment
            pt_b (np.ndarray): end point of segment
            color (str): color of segment
        """
        cv2.line(
            image,
            pt_a,
            pt_b,
            color,
            5
        )
        return image
