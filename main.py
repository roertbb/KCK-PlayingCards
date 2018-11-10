import os
from contour_detection import findContours


def load_image(dirname):
    images = os.listdir(dirname)
    for image in images:
        findContours(dirname+image)


load_image('./test-data/')
