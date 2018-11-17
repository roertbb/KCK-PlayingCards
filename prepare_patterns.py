import os
import cv2


images = os.listdir('patterns_unprepared')
for image in images:
    img = cv2.imread('./patterns_unprepared/'+image)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(grayscale_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, 2)
    thresh = (255 - thresh)

    thresh = cv2.resize(thresh, (30, 60))

    cv2.imwrite('./patterns/'+image, thresh)