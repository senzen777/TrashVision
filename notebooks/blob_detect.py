#!/usr/bin/env python3
import cv2
import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter, gaussian_filter
from scipy import signal
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import sobel, rotate
import random
import sys
import os

verbose = False

"""Hyperparameter List"""

# detect_blobs
scale_factor = 2 ** (1/16) # difference between adjacent images in scale space
scale_size = 1 # how many scales to do
scale_start = 20
num_orientations = 36

# Extra Credit
def detect_blobs(image):
  """Laplacian blob detector.

  Args:
  - image (2D float64 array): A grayscale image.

  Returns:
  - corners (list of 2-tuples): A list of 2-tuples representing the locations
      of detected blobs. Each tuple contains the (x, y) coordinates of a
      pixel, which can be indexed by image[y, x].
  - scales (list of floats): A list of floats representing the scales of
      detected blobs. Has the same length as `corners`.
  - orientations (list of floats): A list of floats representing the dominant
      orientation of the blobs.
  """

  X = image.shape[1]
  Y = image.shape[0]

  # generate normalized laplacian of gaussians scale space
  if (verbose):
      print('\tFinding DoGs')
  gausses = np.zeros((Y,X,scale_size+1))
  DoGs = np.zeros((Y,X,scale_size))
  for i in range(scale_size+1):
      sig = scale_start * (scale_factor ** i)
      gausses[:,:,i] = gaussian_filter(image, sigma=sig, mode='reflect')
      if (i > 0):
          DoGs[:,:,i-1] = gausses[:,:,i] - gausses[:,:,i-1]

  # find local extrema in DoGs
  if (verbose):
      print('\tFinding maxima')
  maxs = maximum_filter(DoGs,3)
  mins = minimum_filter(DoGs,3)
  extrema = np.copy(DoGs)
  extrema[np.logical_not(np.logical_or(extrema == maxs, extrema == mins))] = 0

  # calculate principal orientations and throw out edges
  if (verbose):
      print('\tFinding orientations')
  candidates = np.argwhere(extrema > 0)
  corners = []
  scales = []
  orientations = []
  for ind in candidates:
      row = ind[0]
      col = ind[1]
      sig = ind[2]
      scale = scale_start * (scale_factor ** sig)

      half_window = int(scale / 2) # make a window of size scale around the candidate
      # throw out all too close to the edge
      if col - half_window < 0:
          continue
      elif col + half_window > X:
          continue
      elif row - half_window < 0:
          continue
      elif row + half_window > Y:
          continue

      slice = DoGs[row - half_window:row + half_window, col - half_window:col + half_window, sig]

      gauss = weight_gaussian(half_window * 2, scale * 1.5)
      # calculate gradient
      sobelX = sobel(slice, axis=1)
      sobelY = sobel(slice, axis=0)
      slice_magnitudes = np.sqrt(np.square(sobelX) + np.square(sobelY))
      slice_directions = np.arctan2(sobelY, sobelX)

      # gaussian weight magnitudes
      slice_magnitudes = np.multiply(slice_magnitudes, gauss)

      # vote on directions
      slice_magnitudes = np.ravel(slice_magnitudes)
      slice_directions = np.ravel(slice_directions)
      dirs = np.linspace(-1*np.pi, np.pi, num_orientations)
      hist = np.zeros(num_orientations)
      for i in range(len(slice_magnitudes)):
          # find closest histogram bin
          bin = np.abs(dirs - slice_directions[i]).argmin()
          hist[bin] += slice_magnitudes[i]
      best = np.argmax(hist)

      # create additional blobs for all other orientations with high value
      for i in range(len(hist)):
          if hist[i] > hist[best] * 0.9:
              corners.append((col, row))
              scales.append(scale)
              orientations.append(dirs[i])

  return corners, scales, orientations


"""Simple function to return a Gaussian kernel in a square matrix"""
def weight_gaussian(windowsize, sigma):
    gauss = np.zeros((windowsize,windowsize))
    center = float(windowsize / 2)
    for x in range(windowsize):
        for y in range(windowsize):
            exponent = -((((x-center)**2) / (2*(sigma ** 2))) + (((y-center)**2) / (2*(sigma ** 2))))
            gauss[y,x] = np.exp(exponent)
    return gauss

"""Finds one of the largest blobs and scales that to a standardized size"""
def crop_and_rescale(img, blobs, scales):

    h1, w1, c1 = img.shape

    # Grabs blob largest blob in list
    max_idx = np.argmax(scales)
    max_scale = scales[max_idx]
    max_blob = blobs[max_idx]
    delta = int(max_scale)

    # Checks to see if the bounds of the crop are within the image frame
    left = max_blob[0] - delta
    if left < 0:
        left = 0
    right = max_blob[0] + delta
    if right > w1:
        right = w1 - 1
    top = max_blob[1] - delta
    if top < 0:
        top = 0
    bottom = max_blob[1] + delta
    if bottom > h1:
        bottom = h1 - 1

    img_cropped = img[top:bottom, left:right, :].copy()

    h2, w2, c2 = img_cropped.shape

    # If the image is not square, it makes it square by padding with zeros
    if h2 != w2:
        maxDim = int(max(h2, w2))
        temp = np.zeros((maxDim, maxDim, 3))
        img_cropped = img_cropped/255.0
        temp[0:h2, 0:w2, :] = img_cropped.copy()
        img_cropped = temp.copy()

    # Resizes the image to 256 by 256
    img_resized = cv2.resize(img_cropped, (256, 256))

    return img_resized

if __name__ == '__main__':
    path = "data/TrashVision Test Set/Compost"

    jpg_files = [f for f in os.listdir(path) if f.endswith('.jpg')]

    os.chdir(path)

    if verbose:
      print('Running Extra Credit Main')

    for f in range(len(jpg_files)):
        img_path1 = jpg_files[f]
        print(img_path1)

        img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) / 255.0

        rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        print('Detecting Blobs')
        corners1, scales1, orientations1 = detect_blobs(gray1)
        print('Found '+str(len(corners1)))

        cropped_and_rescaled_im1 = crop_and_rescale(rgb1, corners1, scales1)
        plt.imsave(img_path1+'cropped.png', cropped_and_rescaled_im1)
