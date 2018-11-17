import numpy as np
import cv2
from contour_detection import preprocess_threshhold
from hu_moments import cards_hu_moments

WIDTH = 378
HEIGHT = 534


def fetch_card(img, contours):
    h = np.float32([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
    cards = []
    for (approx, contour) in contours:
        transform = cv2.getPerspectiveTransform(np.float32(approx), h)
        warp = cv2.warpPerspective(img, transform, (WIDTH, HEIGHT))
        thresh = preprocess_threshhold(warp)

        thresh = rotate_image_properly(thresh)

        moments = cv2.moments(thresh)
        hu_moments = cv2.HuMoments(moments).flatten()

        best_fit = calc_best_fit(hu_moments)

        cards.append((contour, best_fit))
    return cards


def rotate_image_properly(thresh):

    # check in which corner is the symbol - our Region Of Interest
    AREA1 = 140 * 40
    AREA2 = 65 * 90
    left = thresh[15:155, 0:40]
    right = thresh[15:155, 327:367]
    upper_left = thresh[5:70, 15:105]
    upper_right = thresh[10:75, 280:370]
    edges = [cv2.countNonZero(left)/AREA1, cv2.countNonZero(
        right)/AREA1, cv2.countNonZero(upper_left)/AREA2, cv2.countNonZero(upper_right)/AREA2]
    index = edges.index(min(edges))

    if index == 1:
        thresh = cv2.flip(thresh, +1)
    elif index == 2:
        thresh = cv2.transpose(thresh)
        thresh = cv2.resize(thresh, (0, 0), fx=WIDTH /
                            HEIGHT, fy=HEIGHT/WIDTH)
        thresh = cv2.flip(thresh, -1)
    elif index == 3:
        thresh = cv2.transpose(thresh)
        thresh = cv2.resize(thresh, (0, 0), fx=WIDTH /
                            HEIGHT, fy=HEIGHT/WIDTH)
        thresh = cv2.flip(thresh, 1)

    # cv2.imshow('image', thresh)
    # cv2.waitKey(0)

    return thresh


def calc_best_fit(hu_moments):
    min_dist = 99999999999
    key = "undefined"
    for card in cards_hu_moments:
        dist = np.linalg.norm(hu_moments - cards_hu_moments[card])
        if (dist < min_dist):
            min_dist = dist
            key = card

    return key
