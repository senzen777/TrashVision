#!/usr/bin/env python3
import cv2
import numpy as np
from scipy import signal
from scipy import ndimage

def blob_detect(image):

    """numSigmas = 7
    sigma0 = 12
    s = 2
    k = np.linspace(1, 16, numSigmas)"""

    # Hyperparameters
    numSigmas = 48
    scale_factor = 2 ** (1/16)
    scale_start = 12

    # Initialize numpy arrays that will hold gausses and DoGs
    gauss = np.zeros((image.shape[0], image.shape[1], numSigmas))
    DoG = np.zeros((image.shape[0], image.shape[1], numSigmas - 1))

    for i in range(numSigmas):

        # Calculate sigma
        #sigma = sigma0*s**k[i]
        sigma = scale_start * (scale_factor ** i)

        # If first iteration, calculate gauss, otherwise calculate gauss and DoG
        if i == 0:
            gauss[:,:,i] = ndimage.gaussian_filter(image, sigma, mode="reflect")
        else:
            gauss[:,:,i] = ndimage.gaussian_filter(image, sigma, mode="reflect")
            DoG[:,:,i-1] = gauss[:,:,i] - gauss[:,:,i-1]

    # Find the extrema and set everything else equal to zero
    maxes = ndimage.maximum_filter(DoG, size=3)
    mins = ndimage.minimum_filter(DoG, size=3)
    extrema = np.copy(DoG)
    extrema[np.logical_not(np.logical_or(extrema == maxes, extrema == mins))] = 0

    # Extract the blobs from the extrema data
    blobs = []
    outputs = np.argwhere(extrema > 0)
    x = outputs[0]
    y = outputs[1]
    sig = outputs[2]

    for i in range(len(y)):
        blobs.append(((x[i], y[i]), int(sig[i])))

    return blobs

def baseline_main():
    img = cv2.imread('test3.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blobs = blob_detect(gray)
    print(blobs)
    for i in range(len(blobs)):
        img_blob = cv2.circle(img, blobs[i][0], blobs[i][1], (0, 0, 255), 1)

    cv2.imwrite("img_blob.png", img_blob)

if __name__ == '__main__':
    baseline_main()
