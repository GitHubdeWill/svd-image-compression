import cv2
import numpy as np
import sys


# Convert to greyscale no matter what
def forceGrey(img):
    if len(img.shape) == 2:
        return img
    elif len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# A simple helper function for image showing
def show(img):
    cv2.imshow(".", img)
    cv2.waitKey()


# Print iterations progress
def printProgressBar(iteration, total, prefix='>progress: ', suffix='complete',
                     decimals=1, length=50, fill='#'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    # Print New Line on Complete
    if iteration == total:
        print ""


def findPeak(xs, ys, count=15):

    peaks = []
    print ">>Looking for peaks:"

    lastP = [xs[0], 1]  # index, height
    temp_low = 1
    for i in range(1, len(ys)):

        # extract peaks
        if (ys[i] - ys[i-1] < 0 and lastP[0] == xs[i-1]):
            peaks.append(lastP)
            temp_low = ys[i]
        elif ys[i] - ys[i-1] < 0:
            temp_low = ys[i]
            continue
        else:
            lastP = [xs[i], ys[i]-temp_low]
    if len(peaks) == 0:
        return None

    peaks = np.array(peaks)

    x = peaks[np.argsort(peaks[:, 1], axis=0)]

    # print x
    if count > len(peaks)-1:
        count = len(peaks)-1

    ret = []
    for i in range(count):
        ret.append(x[-1-i][0])

    return ret
