import cv2
import cv2 as cv
import numpy as np

RED = (0,0,255)
BLUE = (255,0,0)
GREEN = (0,255,0)
TEAL = (255,255,0)
WHITE = (255,255,255)
BLACK = (0,0,0)

X = 640
Y = 480


def wait_or_quit():
    while True:
        key = cv2.waitKey(8000000)
        if key == ord('q'):
            exit(0)

def min_dist_in_set(points: list) -> list:
    """ Returns the two points with the minimal euclidean norm """
    if len(points) < 2:
        return None

    min_dist = float("inf")
    best = None

    for i in range(len(points)):
        for j in range(i+1, len(points)):
            a = np.array(points[i])
            b = np.array(points[j])

            if cv.norm(a-b) < min_dist:
                min_dist = cv.norm(a-b)
                best = [a, b]

    return best


def process(image: np.ndarray, low_hsv: list, high_hsv: list, y: int, x: int):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low_hsv, high_hsv)

    image[mask == 0] = BLACK
    image[mask != 0] = WHITE

    if is_show_mask:
        cv2.imshow(f"Masked image {y} method {x}", image)

    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    conts, _ = cv2.findContours(image_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(conts) > 0:
        conts_sort = sorted(conts, key=lambda x: -cv2.contourArea(x))

        for c in conts_sort[:1]:
            rect = cv2.minAreaRect(c)
            box = np.intp(cv2.boxPoints(rect))
            cv2.drawContours(image, [box], 0, RED, 2)

            ext_left = tuple(c[c[:, :, 0].argmin()][0])
            ext_right = tuple(c[c[:, :, 0].argmax()][0])
            ext_top = tuple(c[c[:, :, 1].argmin()][0])
            ext_bot = tuple(c[c[:, :, 1].argmax()][0])

            cv2.drawContours(image, [c], 0, GREEN, 1)
            cv2.circle(image, ext_left, 8,  RED, -1)
            cv2.circle(image, ext_right, 8, RED, -1)
            cv2.circle(image, ext_top, 8,   RED, -1)
            cv2.circle(image, ext_bot, 8,   RED, -1)

            M = cv2.moments(c)
            cx = int(M['m10']/(M['m00'] or 1))
            cy = int(M['m01']/(M['m00'] or 1))

            cv2.circle(image, (cx, cy), 8,  RED, -1)

            points = [ ext_left, ext_right, ext_top, ext_bot ]
            a, b = min_dist_in_set(points)

            if a[0] < b[0]:
                a, b = b, a
            unit = (a-b) / cv.norm(a-b)
            scaled = np.intp(unit * 7 * cy//9) + np.array([cx,cy])
            cv2.line(image, (cx,cy), scaled, TEAL, 3)
            cv2.line(image, a, b, TEAL, 3)

    if is_show_conts:
        cv2.imshow(f"Contours method {y} image {x}", image)


if __name__ == '__main__':
    image_set = ['1677789529.png', '1677789713.png','1677789529.png']
     #'./1677625016.png', './1677625035.png']

    is_show_mask = False
    is_show_conts = True

    for i, image_str in enumerate(image_set):
        image = cv2.imread(image_str)
        ix = image.shape[0]
        image = image[ix//3:9*ix//10, :]
        #image = image[ix//3:5*ix//6, :]

        #process(image, (22, 79, 147), (41, 161, 215), 1, i+1)
        # Alternative masking (better?)
        # Good masks for right-hand-side: (21, 15, 134), (66, 255, 255)
        #process(image, (21, 15, 134), (66, 255, 255), 2, i+1)
        #
        # This masking removes the bottom of apriltags too
        process(image, (13, 70, 170), (32, 135, 255), 3, i+1)

    wait_or_quit()
    exit(0)

# https://pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
# https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
