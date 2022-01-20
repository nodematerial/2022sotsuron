
import os
import sys
import cv2
import warnings
sys.path.append(os.path.abspath("../../.."))
from utils import *

warnings.filterwarnings('ignore')

gaussian = 5
brightness = 9
kernel = 3
iteration = 4
blocksize = 31
inverse = True
unit = "nm"
ratio = 4.143646408839779
cutting = [200, 230, 7, 0]
bw = 0.06

LOGGER.info('[start program]')
img, top, bottom = preprocessing('no2.tif', gaussian = gaussian, cutting = cutting)
markers, borderless_markers, size, centroids = get_markers(img, kernel = kernel,
        blocksize=blocksize, brightness = brightness, iteration = iteration, inverse = inverse)
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
