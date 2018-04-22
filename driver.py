import cv2
import numpy as np
import sys
import os

import imgutils


def main():

    img = cv2.imread(sys.argv[1])
    img = imgutils.forceGrey(img)

    img = cv2.resize(img, (100, 100))

    cv2.imwrite(sys.argv[1].split(".")[0] + "100." + sys.argv[1].split(".")[1], img)



main()
