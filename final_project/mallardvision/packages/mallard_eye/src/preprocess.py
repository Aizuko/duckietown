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
        return None

    corners = corners.squeeze()

    # sort corners: top left, top right, bottom right, bottom left
    ascending_y_indices = np.argsort(corners[:, 1])
    corners = corners[ascending_y_indices]
    ascending_x_indices = np.argsort(corners[:2, 0])
    corners[:2] = corners[:2][ascending_x_indices]
    descending_x_indices = np.argsort(corners[2:, 0])[::-1]
    corners[2:] = corners[2:][descending_x_indices]
    return corners


def warp_image(image: np.ndarray):
    """
    Warp the image to obtain a 28x28 mask of the digit.

    Args:
        image: raw image

    Returns:
        image_mask: warped image of the digit
    """
    print("Staring warpd")
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
    contours, _ = cv.findContours(
        image_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv.contourArea(
        contour) > params["min_note_corner_area_blue"]]
    if len(contours) < 1:
        print("warning: no contours found")
        return None, None

    largest_contour = sorted(
        contours,
        key=lambda x: cv.contourArea(x),
        reverse=True)[0]
    convex_hull = cv.convexHull(largest_contour)

    corners = get_corners(convex_hull)
    if corners is None:
        print("warning: not 4 corners found")
        return None, None

    corners = corners.astype(np.float32)
    warped_corners = np.array(
        [[0, 0], [28, 0], [28, 28], [0, 28]], dtype=np.float32
    )

    H = cv.getPerspectiveTransform(corners, warped_corners)

    image_warped = cv.warpPerspective(image, H, (28, 28))

    image_hsv = cv.cvtColor(image_warped, cv.COLOR_BGR2HSV)
    image_mask = cv.inRange(
        image_hsv,
        params["low_hsv_black"],
        params["high_hsv_black"])

    return image_mask, corners


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
