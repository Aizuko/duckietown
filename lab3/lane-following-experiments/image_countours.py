import cv2
import numpy as np


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

    if len(white_conts) > 0:
        c = max(white_conts, key=cv2.contourArea)
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.line(white_channel, (cx,0),(cx,720),  (0,255,0),1)
        cv2.line(white_channel, (0,cy),(1280,cy), (0,255,0),1)
        cv2.drawContours(white_channel, white_conts, -1, (0,255,0), 1)

    yellow_grey = cv2.cvtColor(yellow_channel, cv2.COLOR_BGR2GRAY)
    yellow_conts, _ = cv2.findContours(yellow_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(yellow_conts) > 0:
        c = max(yellow_conts, key=cv2.contourArea)
        M = cv2.moments(c)
        cx = int(M['m10']/(M['m00'] or 1))
        cy = int(M['m01']/(M['m00'] or 1))
        cv2.line(yellow_channel, (cx,0),(cx,720), (0,255,0),1)
        cv2.line(yellow_channel, (0,cy),(1280,cy),(0,255,0),1)
        cv2.drawContours(yellow_channel, yellow_conts, -1, (0,255,0), 1)

    #white_channel = cv2.drawContours(white_channel, white_conts, -1, (0,255,0), 3)

    cv2.imshow('white', white_channel)
    cv2.waitKey(8000)
    #cv2.imshow('red', red_channel)
    #cv2.waitKey(8000)
    cv2.imshow('yellow', yellow_channel)
    cv2.waitKey(8000)
    #cv2.imshow('raw', image)
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


def another_cont(image):
    yellow_channel = mask_range_rgb(image.copy(), [100, 40, 0], [240, 255, 80], [255]*3)
    yellow_channel = mask_range_rgb(yellow_channel, [0]*3, [254]*3, [0]*3)

    yellow_grey = cv2.cvtColor(yellow_channel, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(yellow_grey, 127, 255, 0)
    a, b = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = a[4]
    #cv2.drawContours(yellow_channel, [cnts], 0, (0,255,0), 3)

    conts_sort = sorted(a, key=lambda x: -cv2.contourArea(x))

    # https://stackoverflow.com/questions/7263621/how-to-find-corners-on-a-image-using-opencv


    for i in range(100):
        c = conts_sort[i]
        rect = cv2.minAreaRect(c)
        print(rect)
        cv2.drawContours(yellow_channel, [c], 0, (0,255,0), 3)

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(yellow_channel, [box], 0, (0,255,0), 1)
        #cv2.rectangle(yellow_channel, (int(rect[0][0]), int(rect[0][1])), (int(rect[1][0]), int(rect[1][1])), (0,255,0))

        M = cv2.moments(c)
        cx = int(M['m10']/(M['m00'] or 1))
        cy = int(M['m01']/(M['m00'] or 1))

        print(cx,cy)
        cv2.line(yellow_channel, (cx,0),(cx,y), (0,255,0),1)
        cv2.line(yellow_channel, (0,cy),(x,cy),(0,255,0),1)

        cv2.imshow('cont', yellow_channel)
        key = cv2.waitKey(8000)
        if key == ord('q'):
            break

if __name__ == '__main__':
    image = cv2.imread('screenshot_shm.png')
    y, x, _ = image.shape

    image = image[y//3:2*y//3, :]  # Middle third
    cv2.line(image, (x//2, 0), (x//2, y-1), (0,254,0), thickness=2)

    #image = image[60:-20, :-220]  # From fullscreen waybook view


    another_cont(image)
    exit(0)
    channel_masking(image)
    canny_detection(image)
    cv2.waitKey(8000)
