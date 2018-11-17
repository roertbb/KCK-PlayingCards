import numpy as np
import cv2

CHOOSEM_CONTOURS_NUM = 200
CARDS_ALPHA = 0.2


def preprocess_threshhold(img):
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # add some blur in order to reduce noise
    grayscale_img = cv2.blur(grayscale_img, (5, 5))
    # grayscale_img = cv2.medianBlur(grayscale_img, 7)
    thresh = cv2.adaptiveThreshold(grayscale_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, 2)
    # perform some opening in order to reduce noise
    kernel = np.ones((3, 3))
    for i in range(2):
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return thresh


def find_contours(img):
    height, width = img.shape[:2]
    thresh = preprocess_threshhold(img)
    # detect contours
    contours_detected = cv2.Canny(thresh, 50, 250)
    _, contours, _ = cv2.findContours(
        contours_detected, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours descending by their area
    index_sort = sorted(
        range(len(contours)), key=lambda i: cv2.contourArea(contours[i]), reverse=True)
    choosen_contours = [contours[i]
                        for i in index_sort[0:CHOOSEM_CONTOURS_NUM]]

    contours_map = np.ones((height, width))
    card_contours = []
    for c in choosen_contours:
        perimeter = 0.01*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, perimeter, True)
        # skip when it's not an rectangle
        if len(approx) != 4:
            continue
        # check if contour isn't nested in another one
        copied_map = np.copy(contours_map)
        cv2.drawContours(copied_map, [c], -1,
                         (0, 255, 0), thickness=cv2.FILLED)
        # if so, just skip it
        if np.array_equal(copied_map, contours_map):
            continue
        # if no, add to "map" of already added contours
        contours_map = copied_map

        card_contours.append((approx, c))

    return card_contours


def highlight_detected(img, cards):
    for (contour, match) in cards:
        # create transparent background
        overlay = img.copy()
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), -1)
        cv2.addWeighted(overlay, CARDS_ALPHA, img, 1 - CARDS_ALPHA, 0, img)
        # add notmal contour
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
        # calculate center if contour => get position for text
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(img, match, (cX-(len(match)*5), cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('image', img)
    cv2.waitKey(0)
