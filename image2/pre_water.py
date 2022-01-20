import cv2
import os
import sys
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.abspath(".."))
from utils import colorizer2, cutter

gaussian = 3
brightness = -11
kernel = 3
iteration = 2
blocksize = 21
a = 100
b = 400
c = 220
d = 520

img = cv2.imread('image2.tif')
img = cv2.GaussianBlur(img, (gaussian, gaussian), 0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bin_img = cv2.adaptiveThreshold(
gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blocksize, brightness)

kernel = np.ones((kernel, kernel), np.uint8)
opening = cv2.morphologyEx(
    bin_img, cv2.MORPH_OPEN, kernel, iterations=iteration)
sure_bg = cv2.dilate(opening, kernel, iterations=iteration)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(
    dist_transform, 0.1*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
ret, markers, stats, centroids1 = cv2.connectedComponentsWithStats(
    sure_fg, connectivity=4)
for i in tqdm(range(ret)):
    if stats[i, 4] < 40:
        markers = np.where(markers == i, 0, markers)
markers = markers+1
markers[unknown == 255] = 0
markers = cv2.watershed(img, markers)
color_makrers = colorizer2(markers)

np.set_printoptions(threshold=10000)

cv2.imwrite('pre&watershed.jpg', markers)
cv2.imwrite('pre&watershed_color.jpg', color_makrers)
cv2.imwrite('pre&watershed_color_cut.jpg', cutter(a,b,c,d,color_makrers))

