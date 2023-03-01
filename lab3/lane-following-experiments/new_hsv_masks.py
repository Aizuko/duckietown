import cv2
import numpy as np


def wait_or_quit():
    while True:
        key = cv2.waitKey(8000000)
        if key == ord('q'):
            exit(0)

if __name__ == '__main__':
    image_set = ['./1677625016.png', './1677625035.png']

    for i, image_str in enumerate(image_set):
        image = cv2.imread(image_str)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (22, 79, 147), (41, 161, 215))

        image[mask == 0] = [0]*3
        image[mask != 0] = [255]*3
        cv2.imshow(str(i), image)

        image = cv2.imread(image_str)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (21, 15, 134), (66, 255, 255))  # Better mask

        image[mask == 0] = [0]*3
        image[mask != 0] = [255]*3
        cv2.imshow(str(i) + 'two', image)

    wait_or_quit()
    exit(0)
