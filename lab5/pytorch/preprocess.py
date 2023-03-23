import glob
from pathlib import Path

import cv2 as cv
import numpy as np


def get_corners(
        convex_hull: np.ndarray,
        k: int = 4,
        left_epsilon: float = 0,
        right_epsilon: float = 100):
    """
    Get the corners of the convex hull using the Douglas-Peucker algorithm,
    applying binary search to find the epsilon value that gives k corners.

    Args:
        convex_hull: convex hull of the contour

    Returns:
        corners: array of shape (k, 2) containing the corners
    """
    while left_epsilon < right_epsilon:
        epsilon = (left_epsilon + right_epsilon) / 2
        corners = cv.approxPolyDP(convex_hull, epsilon, True)
        if len(corners) == k:
            break
        elif len(corners) < k:
            right_epsilon = epsilon
        else:
            left_epsilon = epsilon
    else:
        return

    corners = corners.squeeze()

    # sort corners: top left, top right, bottom right, bottom left
    ascending_y_indices = np.argsort(corners[:, 1])
    corners = corners[ascending_y_indices]
    ascending_x_indices = np.argsort(corners[:2, 0])
    corners[:2] = corners[:2][ascending_x_indices]
    descending_x_indices = np.argsort(corners[2:, 0])[::-1]
    corners[2:] = corners[2:][descending_x_indices]
    return corners


def warp_image(image: np.ndarray, visualization_path: Path = None):
    """
    Warp the image to obtain a 28x28 mask of the digit.

    Args:
        image: raw image
        visualization_path: path to save the visualization

    Returns:
        image_mask: warped image of the digit
    """
    params = {
        "low_hsv_blue": (100, 80, 80),
        "high_hsv_blue": (110, 255, 255),
        "low_hsv_black": (0, 0, 0),
        "high_hsv_black": (180, 255, 90),
        "min_note_corner_area_blue": 100
    }

    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    image_mask = cv.inRange(
        image_hsv,
        params["low_hsv_blue"],
        params["high_hsv_blue"])
    if visualization_path is not None:
        cv.imwrite(str(visualization_path / "mask1.png"), image_mask)

    contours, _ = cv.findContours(
        image_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv.contourArea(
        contour) > params["min_note_corner_area_blue"]]
    if len(contours) < 1:
        print("warning: no contours found")
        return

    largest_contour = sorted(
        contours,
        key=lambda x: cv.contourArea(x),
        reverse=True)[0]
    if visualization_path is not None:
        image_copy = image.copy()
        cv.drawContours(
            image_copy, [largest_contour], -1, (0, 255, 255), 2)
        cv.imwrite(str(visualization_path / "contour.png"), image_copy)

    convex_hull = cv.convexHull(largest_contour)
    if visualization_path is not None:
        image_copy = image.copy()
        cv.drawContours(
            image_copy, [convex_hull], -1, (0, 255, 255), 2)
        cv.imwrite(str(visualization_path / "convex_hull.png"), image_copy)

    corners = get_corners(convex_hull)
    if visualization_path is not None:
        image_copy = image.copy()
        cv.drawContours(
            image_copy, [corners], -1, (0, 255, 255), 2)
        cv.imwrite(str(visualization_path / "corners.png"), image_copy)

    if corners is None:
        print("warning: not 4 corners found")
        return

    corners = corners.astype(np.float32)
    warped_corners = np.array(
        [[0, 0], [28, 0], [28, 28], [0, 28]], dtype=np.float32
    )

    H = cv.getPerspectiveTransform(corners, warped_corners)

    image_warped = cv.warpPerspective(image, H, (28, 28))

    if visualization_path is not None:
        cv.imwrite(str(visualization_path / "warp.png"), image_warped)

    image_hsv = cv.cvtColor(image_warped, cv.COLOR_BGR2HSV)
    image_mask = cv.inRange(
        image_hsv,
        params["low_hsv_black"],
        params["high_hsv_black"])

    return image_mask


def preprocess_image(image: np.ndarray):
    """
    Remove 2px from the left and right borders.

    Args:
        image: image of the digit

    Returns:
        image: preprocessed image
    """
    image[:, :2] = 0
    image[:, -2:] = 0
    return image


def normalize_image(image: np.ndarray, mean=(0.1307,), std=(0.3081,)):
    """
    Normalize the image to be a float32 array between 0 and 1.

    Args:
        image: image of the digit

    Returns:
        image: normalized image
    """
    image = image.astype(np.float32)
    image /= 255
    image -= mean
    image /= std
    return image


def main():
    CUSTOM_DATA_DIR = Path("data/custom_data")
    PROCESSED_DATA_DIR = Path("data/processed_data")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DATA_DIR = Path("data/debug_data")
    DEBUG_DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = glob.glob(str(CUSTOM_DATA_DIR / "**/*.jpeg"))
    paths = [Path(path) for path in paths]

    for read_path in paths:
        image = cv.imread(str(read_path))
        debug_image = image.copy()
        image = warp_image(image)
        if image is None:
            print(f"warning: {read_path} not processed")
            continue
        image = preprocess_image(image)

        # embed preprocessed image in debug image
        debug_image[:32, :32] = np.repeat(cv.resize(image, (32, 32))[
                                          :, :, np.newaxis], 3, axis=2)
        debug_write_path = DEBUG_DATA_DIR / read_path.relative_to(
            CUSTOM_DATA_DIR)
        debug_write_path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(debug_write_path), debug_image)

        # save preprocessed image
        write_path = PROCESSED_DATA_DIR / read_path.relative_to(
            CUSTOM_DATA_DIR).with_suffix(".png")
        write_path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(write_path), image)


if __name__ == "__main__":
    main()
