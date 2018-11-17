import os
import numpy as np
import cv2
from contour_detection import preprocess_threshhold
from hu_moments import cards_hu_moments

WIDTH = 378
HEIGHT = 534


def get_cards_from_image(img, contours):
    h = np.float32([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
    cards = []
    for (approx, contour) in contours:
        # transform image to rectangle
        transform = cv2.getPerspectiveTransform(np.float32(approx), h)
        warp = cv2.warpPerspective(img, transform, (WIDTH, HEIGHT))
        thresh = preprocess_threshhold(warp)
        # rotate when transformation above rotated it
        thresh = rotate_image_properly(thresh)
        # calculate it's moments - hu moments seems to bo not so nice to work with - it seems to be next random algorithm :)
        # moments = cv2.moments(thresh)
        # hu_moments = cv2.HuMoments(moments).flatten()
        # get best fit according to moments set
        # best_fit = calc_best_card_fit(hu_moments)
        best_fit = predict_card(thresh)

        cards.append((contour, best_fit))

    return cards


def predict_card(thresh):
    value_patterns = load_value_patterns()
    value = predict_value(thresh, value_patterns)

    return value


def load_value_patterns():
    value_patterns = {}
    # load patterns from pattern directory
    patterns = os.listdir('patterns')
    for pattern in patterns:
        pattern_img = cv2.imread('./patterns/'+pattern, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(pattern_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 15, 2)
        # parse name: loweracase => uppercase, 0 => 10 (detecting 0 - one contour bigger than 1)
        pattern_name = pattern.split('.')[0].upper()
        if pattern_name == '0':
            pattern_name = '10'

        # store that value in dictionary
        value_patterns[pattern_name] = thresh

    return value_patterns


def predict_value(thresh, value_patterns):
    # choose left upper corner in order to get cards value
    left = thresh[15:100, 0:70]
    # remove noise and add some blur
    left = cv2.medianBlur(left, 7)
    # black on white => white on black
    left = (255-left)
    # get contour
    _, contours, _ = cv2.findContours(
        left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # filter contours, which area is bigger than 2% of whole  area
    a = 0.02 * 85*70
    choosen_contours = list(filter(lambda c: cv2.contourArea(c) > a, contours))
    # when no proper contours find
    if len(choosen_contours) < 1:
        return "undefined"
    # get bounding rect of biggest contour, which is probably our value
    x, y, w, h = cv2.boundingRect(choosen_contours[0])
    # get what's inside bounding rect
    num = left[y:y+h, x:x+w]
    # resize it to pattern size
    num = cv2.resize(num, (30, 60))

    # check how "similar" our num is to pattern
    value_fit = {}
    for (key, value) in value_patterns.items():
        # calculate different pixels and store that difference
        value_fit[key] = cv2.countNonZero(cv2.absdiff(num, value))

    # choose best fit = the smallest difference
    best_fit = min(value_fit, key=value_fit.get)

    # debugging => uncomment to take a look what was found
    # print(best_fit, value_fit)
    # cv2.imshow('image', num)
    # cv2.waitKey(0)

    # return value we predicted
    return best_fit


def rotate_image_properly(thresh):

    # check in which corner is the symbol
    AREA1 = 140 * 40
    AREA2 = 65 * 90
    # left
    left = thresh[15:155, 0:40]
    # right - card flipped
    right = thresh[15:155, 327:367]
    # left-upper, card id detected horizontal
    upper_left = thresh[5:70, 15:105]
    # right-upper, card id detected horizontal
    upper_right = thresh[10:75, 280:370]

    # find card position according to percentage of black pixels in cut out image
    edges = [cv2.countNonZero(left)/AREA1, cv2.countNonZero(
        right)/AREA1, cv2.countNonZero(upper_left)/AREA2, cv2.countNonZero(upper_right)/AREA2]
    index = edges.index(min(edges))

    # if iamge is rotated, rotate it to desirez orientation
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


# def calc_best_card_fit(hu_moments):
#     # choose best fit => least difference between hu moments
#     min_dist = 99999999999
#     key = "undefined"
#     for card in cards_hu_moments:
#         dist = np.linalg.norm(hu_moments - cards_hu_moments[card])
#         if (dist < min_dist):
#             min_dist = dist
#             key = card

#     return key


# def calc_best_numbers_hu(hu_moments):
#     min_dist = np.inf
#     key = "undefined"
#     for num in numbers_hu:
#         dist = np.linalg.norm(hu_moments - numbers_hu[num])
#         if (dist < min_dist):
#             min_dist = dist
#             key = num

#     return key
