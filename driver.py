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
    img = imgutils.forceGrey(img)

    img = cv2.resize(img, (100, 100))
    img = np.reshape(img, (1, 10000))
    img = np.reshape(img, (100, 100))
    cv2.imshow('he', img)
    cv2.waitKey(0)
    # yeah...
    data_x = img
    # print('X = ', data_x.shape)
    #
    # print('Implement PCA here ...')
    # U, s, V = np.linalg.svd(data_x)
    #
    # cache = [(data_x.reshape(100, 100), "Original")]
    # for i in [3, 5, 10, 30, 50, 100, 150, 300]:
    #     print("k = " + str(i))
    #     w_k = V[:i]
    #     X_projected = np.dot(data_x, w_k.T)
    #     X_recon = np.dot(X_projected, w_k)
    #     print("Reconstruction error: " + str(np.sqrt(np.mean((data_x-X_recon)**2))))
    #     print("Compression rate: " + str((X_projected.nbytes + w_k.nbytes)/data_x.nbytes))
    #     cache.append((X_recon[1].reshape(100, 100), "k=" + str(i)))
    # imgutils.helper_plot_grid(cache, "PCA reconstruction", "gray")

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
                # cv2.imwrite(name+str(c)+"RGB."+file.split(".")[-1], im)

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
