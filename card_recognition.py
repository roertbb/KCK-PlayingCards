import numpy as np
import cv2
from contour_detection import preprocess_threshhold


cards_hu_moments = {
    "6 - club": [8.15204945e-04,  7.28162058e-08,  5.29624079e-14,  3.17606849e-14,
                 1.21555679e-27,  8.49967760e-18, - 4.68238882e-28],
    "K - diamond": [1.11107109e-03,  1.29274758e-07,  2.34392888e-14,  4.80633284e-14,
                    1.61242994e-27,  5.84897136e-18, - 5.04087483e-29],
    "2 - spade": [7.34565166e-04,  5.68757962e-08,  1.09295829e-14,  1.19592060e-14,
                  4.92116014e-29,  2.40370908e-18, - 1.27564021e-28],
    "A - spade": [8.30927252e-04, 7.26320432e-08,  1.39278823e-15, 1.25211548e-14,
                  3.31038896e-29, - 2.91608489e-18,  4.04753839e-29],
    "5 - heart": [7.56437195e-04,  6.76586265e-08,  7.12744942e-15,  6.63988805e-16,
                  3.51032504e-32, - 7.92996532e-20,  1.44404234e-30]
}


def fetch_card(img, contours):
    h = np.float32([[0, 0], [378, 0], [378, 534], [0, 534]])
    cards = []
    for (approx, contour) in contours:
        transform = cv2.getPerspectiveTransform(np.float32(approx), h)
        warp = cv2.warpPerspective(img, transform, (378, 534))

        thresh = preprocess_threshhold(warp)
        moments = cv2.moments(thresh)
        hu_moments = cv2.HuMoments(moments).flatten()

        # print(hu_moments)
        # cv2.imshow('image', thresh)
        # cv2.waitKey(0)

        best_fit = calc_best_fit(hu_moments)

        cards.append((contour, best_fit))
    return cards


def calc_best_fit(hu_moments):
    min_dist = 99999999999
    key = "undefined"
    for card in cards_hu_moments:
        dist = np.linalg.norm(hu_moments - cards_hu_moments[card])
        if (dist < min_dist):
            min_dist = dist
            key = card

    return key
