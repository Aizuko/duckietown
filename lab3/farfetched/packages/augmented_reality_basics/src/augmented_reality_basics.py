import cv2
import numpy as np


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

    def process_image(self, image):
        """Undistorts raw images.
        """
        return

    def ground2pixel(self, ground_coordinates):
        """
        transforms points in ground coordinates (i.e. the robot reference frame)
        to pixels in the image.
        """
        return

    def render_segments(self, segments):
        """Plots the segments from the map files onto the image.
        """
        return

    def draw_segment(
            self,
            image: np.ndarray,
            pt_a: np.ndarray,
            pt_b: np.ndarray,
            color: str):
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
        _, [r, g, b] = self.defined_colors[color]
        cv2.line(
            image,
            pt_a,
            pt_b,
            (b * 255, g * 255, r * 255),
            5
        )
        return image
