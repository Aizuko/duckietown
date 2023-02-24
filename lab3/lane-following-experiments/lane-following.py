import cv2
import numpy as np

cap = cv2.VideoCapture('apriltag.mov')


def get_hsv_mask(hsv: np.ndarray, lower: np.ndarray, upper: np.ndarray):
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def lane_geometry(image: np.ndarray):
    """
    https://www.tutorialspoint.com/detection-of-a-specific-color-blue-here-using-opencv-with-python
    https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    gray_mask = get_hsv_mask(hsv, np.array([0, 18, 0]), np.array([255, 180, 118]))
    white_mask = get_hsv_mask(hsv, np.array([0, 0, 160]), np.array([255, 61, 255]))
    yellow_mask = get_hsv_mask(hsv, np.array([10, 30, 100]), np.array([40, 160, 200]))
    red_mask = get_hsv_mask(hsv, np.array([0, 100, 100]), np.array([20, 255, 255]))
    image[gray_mask > 0] = [128, 128, 128]
    image[white_mask > 0] = [255, 255, 255]
    image[yellow_mask > 0] = [0, 255, 255]
    image[red_mask > 0] = [0, 0, 255]
    return image

def draw_lane_geometry(image, geometry):
    return geometry

while cap.isOpened():
    ret, image = cap.read()
    image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    image = image[118:-10, 30:-220]  # Crop to camera only
    geometry = lane_geometry(image)
    image = draw_lane_geometry(image, geometry)
    cv2.imshow('image', image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
