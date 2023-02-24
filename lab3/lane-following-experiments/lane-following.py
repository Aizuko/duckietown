import cv2
import numpy as np

cap = cv2.VideoCapture('apriltag.mov')


def rgb2bgr(r, g, b):
    return [b, g, r]

def mask_range_rgb(image, lower: list, upper: list, fill: list):
    return mask_range(image, rgb2bgr(*lower), rgb2bgr(*upper), rgb2bgr(*fill))

def mask_range(image, lower: list, upper: list, fill: list):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    image[mask > 0] = fill
    return image

def lane_geometry(image: np.ndarray):
    """
    https://www.tutorialspoint.com/detection-of-a-specific-color-blue-here-using-opencv-with-python
    https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    """
        # White detection
    image = mask_range_rgb(image, [160, 0, 0],   [255, 61, 255], [255]*3)
        # Red detection
    image = mask_range_rgb(image, [130, 100, 0], [250, 250, 20], [255, 0, 0])
        # Yellow detection
    image = mask_range_rgb(image, [100, 40, 0], [240, 255, 80], [255, 255, 0])
        # Black out
    image = mask_range_rgb(image, [0]*3, [200]*3, [0]*3)

    return image

def draw_lane_geometry(image, geometry):
    return geometry

while cap.isOpened():
    ret, image = cap.read()
    image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    #image = image[118:-10, 30:-220]  # Crop to camera only
    image = image[318:-10, 30:-220]  # Crop to lower camera

    white_channel = mask_range_rgb(image.copy(), [160, 0, 0],   [255, 61, 255], [255]*3)
    white_channel = mask_range_rgb(white_channel, [0]*3, [254, 254, 254], [0]*3)
    red_channel = mask_range_rgb(image.copy(), [130, 100, 0], [255, 255, 20], [255, 0, 0])
    red_channel = mask_range_rgb(red_channel, [0]*3, [254, 255, 255], [0]*3)
    yellow_channel = mask_range_rgb(image.copy(), [100, 40, 0], [240, 255, 80], [255, 255, 0])
    yellow_channel = mask_range_rgb(yellow_channel, [0]*3, [254, 254, 255], [0]*3)

    cv2.imshow('white', white_channel)
    cv2.imshow('red', red_channel)
    cv2.imshow('yellow', yellow_channel)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
