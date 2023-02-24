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

def channel_masking(image: np.ndarray):
    white_channel = mask_range_rgb(image.copy(), [160, 0, 0],   [255, 61, 255], [255]*3)
    red_channel = mask_range_rgb(image.copy(), [130, 100, 0], [255, 255, 20], [255]*3)
    yellow_channel = mask_range_rgb(image.copy(), [100, 40, 0], [240, 255, 80], [255]*3)

    white_channel = mask_range_rgb(white_channel, [0]*3, [254]*3, [0]*3)
    red_channel = mask_range_rgb(red_channel, [0]*3, [254]*3, [0]*3)
    yellow_channel = mask_range_rgb(yellow_channel, [0]*3, [254]*3, [0]*3)

    white_grey = cv2.cvtColor(white_channel, cv2.COLOR_BGR2GRAY)
    white_conts, _ = cv2.findContours(white_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    white_channel = cv2.drawContours(white_channel, white_conts, -1, (0,255,0), 3)

    cv2.imshow('white', white_channel)
    #cv2.imshow('red', red_channel)
    #cv2.imshow('yellow', yellow_channel)
    return

def canny_detection(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 255, apertureSize = 3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    if lines is not None:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(image, (x1, y1), (x2, y2), rgb2bgr(0, 255, 0), 3)

    cv2.imshow('uncanny', image)
    return

while cap.isOpened():
    ret, image = cap.read()
    image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    #image = image[118:-10, 30:-220]  # Crop to camera only
    #image = image[318:-10, 30:-230]  # Crop to lower camera
    image = image[208:-20, 40:-240]  # Crop to lower camera tight

    channel_masking(image)
    canny_detection(image)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
