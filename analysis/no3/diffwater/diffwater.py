# -*- coding: utf-8 -*-

import os
import sys
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import size
from numpy.ma.core import count
import seaborn as sns
import warnings
sys.path.append(os.path.abspath("../../.."))
from utils import *

warnings.filterwarnings('ignore')

gaussian = 5
start, end, interval = 8, 17, 1 
kernel = 3
iteration = 3
blocksize = 25
removal = 85
inverse = True
unit = "nm"
ratio = 4.143646408839779
cutting = [190, 255, 7, 0]
bw = 0.06

LOGGER.info('[start program]')
img, top, bottom = preprocessing('no3.tif', gaussian=gaussian, cutting=cutting)
mar, borderless_markers, size, centroids = difference_algo_water(img, 
    kernel=kernel, interval=interval, blocksize=blocksize, start=start, 
    end=end, iteration=iteration,inverse=inverse, removal=removal)

markers = np.zeros_like(img[:, :, 0]).astype(np.int64)
for array in mar:
    for arr in array:
        x, y = arr[0][0], arr[0][1]
        markers[y][x] = -1

H, W = img.shape[0], img.shape[1]
cv2.imwrite('circle.jpg', to_circle(H=H,W=W,size=size, centroids=centroids))
border_mask(markers = markers, img = img)
border_mask_concat(markers = markers, img = img, top = top, bottom = bottom)
border_mask_centroids(img = img, markers = markers, centroids = centroids)
markers_info(size = size, unit = unit, ratio = ratio)
RDF(img = img, bw = bw, centroids = centroids)
RDF(img = img, bw = bw, centroids = centroids, ratio = ratio, unit = unit)
size_dist(size)
size_dist(size, ratio = ratio, unit = unit)
radius_dist(size)
radius_dist(size, ratio = ratio, unit = unit)
cv2.imwrite('colored.jpg', colorizer2(borderless_markers))
