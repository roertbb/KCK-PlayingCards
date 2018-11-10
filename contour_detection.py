import numpy as np
import cv2

CHOOSEM_CONTOURS_NUM = 40


def findContours(image_path):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    height, width = resized_img.shape[:2]
    grayscale_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(grayscale_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, 2)

    # perform some opening in order to reduce noise
    kernel = np.ones((3, 3))
    for i in range(2):
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours_detected = cv2.Canny(thresh, 50, 250)

    _, contours, _ = cv2.findContours(
        contours_detected, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours descending by their area
    index_sort = sorted(
        range(len(contours)), key=lambda i: cv2.contourArea(contours[i]), reverse=True)
    choosen_contours = [contours[i]
                        for i in index_sort[1:CHOOSEM_CONTOURS_NUM]]

    for c in choosen_contours:
        perimeter = 0.01*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, perimeter, True)

        # skip when it's not an rectangle
        if len(approx) != 4:
            continue

        # testing and visualization
        empty = np.ones((height, width))
        cv2.drawContours(empty, [c], -1, (0, 255, 0), thickness=cv2.FILLED)

        cv2.imshow('image', empty)
        cv2.waitKey(0)
