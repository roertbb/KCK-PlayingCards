import numpy as np
import cv2
import os
from contour_detection import find_contours, fetch_card


def load_image(dirname):
    images = os.listdir(dirname)
    for image in images:
        img = cv2.imread(dirname+image)
        resized_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        contours = find_contours(resized_img)
        card_images = fetch_card(resized_img, contours)


load_image('./test-data/')
