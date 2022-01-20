
import os
import sys
import cv2
import numpy as np
import warnings
sys.path.append(os.path.abspath("../../.."))
from utils import *

warnings.filterwarnings('ignore')

gaussian = 1
brightness = 1
kernel = 3
iteration = 2
blocksize = 11
inverse = True
unit = "nm"
ratio = 8.28729281
cutting = [160, 230, 7, 0]
bw = 0.03

LOGGER.info('[start program]')
img, top, bottom = preprocessing('no1.tif', gaussian = gaussian, cutting = cutting)
markers, borderless_markers, size, centroids = get_markers(img, kernel = kernel,
        blocksize=blocksize, brightness = brightness, iteration = iteration, inverse = inverse)
border_mask(markers = markers, img = img)
border_mask_concat(markers = markers, img = img, top = top, bottom = bottom)
border_mask_centroids(img = img, markers = markers, centroids = centroids)
markers_info(size = size, unit = unit, ratio = ratio)
RDF(img = img, bw = bw, centroids = centroids)
RDF(img = img, bw = bw, centroids = centroids, ratio = ratio, unit = unit)
size_dist(size)
size_dist(size, ratio = ratio, unit = unit)
cv2.imwrite('colored.jpg', colorizer2(borderless_markers))

