########################################################################
#
#  Math 545 Final Project
#
#  Professor: Qian-Yong Chen
#  Students: Yi Fung, William He
#
#  Due Date: April 30, 2018
#
#  Note: this is file that helps crop and format the images,
#        preparing the data for SVD later.
#
#  Thank you for grading and have a great summer!
#
########################################################################


import cv2
import numpy as np
import sys
import os
# from scipy import misc
# import imageio
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt


import imgutils


def process(filename):

    img = cv2.imread(filename)
    # img = imgutils.forceGrey(img)

    img = cv2.resize(img, (100, 100))
    # yeah...
    data_x = img

    return img


def main(name):
    # If no name is passed
    if name is None:
        name = sys.argv[1]

    filelist = []

    c = 0

    if os.path.isdir(name):
        # Fix dir name if it's not ended with /
        if not name.endswith("/"):
            name = name+"/"
        filelist = os.listdir(name)

        for file in filelist:
            print(file)

            if file.split('.')[-1] == 'png' or \
               file.split('.')[-1] == 'jpg' or \
               file.split('.')[-1] == 'jpeg':
                c += 1
                print(c)

                key = raw_input("Press any other key to skip, ENTER to continue: ")
                if len(key) > 0:
                    continue

                im = process(name+file)
                cv2.imwrite(name+str(c)+"RGB."+file.split(".")[-1], im)

                print()  # Newline
                raw_input("press any key to continue.")
                print("\n\n\n\n")

    elif os.path.isfile(name):
        if name.split('.')[-1] == 'png' or \
          name.split('.')[-1] == 'jpg' or \
          name.split('.')[-1] == 'jpeg':

            process(name)
            # cv2.imwrite(name+str(c)+"."+file.split(".")[-1], im)

            print()  # Newline
            raw_input("press any key to continue.")
        print("\n\n\n\n")


main(None)
