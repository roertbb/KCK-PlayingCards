import numpy as np
import cv2

CHOOSEM_CONTOURS_NUM = 40


def find_contours(img):
    height, width = img.shape[:2]
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

    # image of already detected contours
    contours_map = np.ones((height, width))

    card_contours = []
    for c in choosen_contours:
        perimeter = 0.01*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, perimeter, True)

        # skip when it's not an rectangle
        if len(approx) != 4:
            continue

        # check if contour already exists
        copied_map = np.copy(contours_map)
        cv2.drawContours(copied_map, [c], -1,
                         (0, 255, 0), thickness=cv2.FILLED)
        if np.array_equal(copied_map, contours_map):
            continue
        contours_map = copied_map

        card_contours.append((approx, c))

    return card_contours


def fetch_card(img, contours):
    h = np.float32([[0, 0], [378, 0], [378, 534], [0, 534]])
    cards = []
    for (approx, contour) in contours:
        transform = cv2.getPerspectiveTransform(np.float32(approx), h)
        warp = cv2.warpPerspective(img, transform, (378, 534))
        cards.append(warp)
        cv2.imshow('image', warp)
        cv2.waitKey(0)
