import numpy as np
import cv2
import os
from contour_detection import find_contours, highlight_detected
from card_recognition import fetch_card


def load_image(dirname):
    images = os.listdir(dirname)
    for image in images:
        img = cv2.imread(dirname+image)
        resized_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        contours = find_contours(resized_img)
        cards = fetch_card(resized_img, contours)
        highlight_detected(resized_img, cards)


load_image('./test-data/')
