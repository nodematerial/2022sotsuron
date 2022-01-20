
import os
import sys
import cv2
import numpy as np
import warnings
sys.path.append(os.path.abspath("../../.."))
from utils import *

warnings.filterwarnings('ignore')

gaussian = 3
start, end = 3, 7
kernel = 3
iteration = 2
blocksize = 13
inverse = True
unit = "nm"
ratio = 8.28729281
cutting = [160, 230, 7, 0]

LOGGER.info('[start program]')
img, top, bottom = preprocessing('no1.tif', gaussian=gaussian, cutting=cutting)
mar, borderless_markers, size, centroids = difference_algo(img, kernel=kernel,
blocksize=blocksize, start=start, end=end, iteration=iteration, inverse=inverse)

markers = np.zeros_like(img[:, :, 0]).astype(np.int64)
for array in mar:
    for arr in array:
        x, y = arr[0][0], arr[0][1]
        markers[y][x] = -1

cv2.imwrite('aaaaaaaa.jpg', colorizer2(borderless_markers))
border_mask(markers = markers, img = img)
border_mask_concat(markers = markers, img = img, top = top, bottom = bottom)
border_mask_centroids(img = img, markers = markers, centroids = centroids)
markers_info(size = size, unit = unit, ratio = ratio)
RDF(img = img, centroids = centroids)
size_dist(size)
cv2.imwrite('colored.jpg', colorizer2(borderless_markers))
