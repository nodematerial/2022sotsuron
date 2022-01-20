import cv2
import os
import sys
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.abspath(".."))
from utils import colorizer, cutter

gaussian = 3
kernel = 3
iteration = 2
blocksize = 21
a = 100
b = 400
c = 220
d = 520

# メイン処理

img = cv2.imread('image2.tif')
img = cv2.GaussianBlur(img, (gaussian, gaussian), 0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
pre_markers = np.zeros_like(gray)
pre_cent_arr = np.zeros_like(gray)
kernel = np.ones((kernel, kernel), np.uint8)

for brightness in range(-29,-6):
    bin_img = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blocksize, brightness)
    opening = cv2.morphologyEx(
        bin_img, cv2.MORPH_OPEN, kernel, iterations=iteration)
    cent_arr = np.zeros_like(opening)
    __, markers, _, centroids = cv2.connectedComponentsWithStats(
        opening, connectivity=4)
    for cent in centroids:
        W, H = int(cent[0]), int(cent[1])
        cent_arr[H][W] = 1

    markers_ = np.zeros_like(markers)
    cent_arr_= np.zeros_like(markers)
    for j in tqdm(range(1,len(centroids)+1)):
        pre_cent = np.where((markers==j) & (pre_cent_arr==1),1,0)
        num_pre_cent_in_marker = np.count_nonzero(pre_cent)
        if num_pre_cent_in_marker >=2:
            markers_ = np.where(markers==j, pre_markers, markers_)
            cent_arr_= np.where(markers==j, pre_cent_arr, cent_arr_)
        else:
            markers_ = np.where(markers==j, markers, markers_)
            cent_arr_= np.where(markers==j, cent_arr, cent_arr_)
    pre_markers = markers_.copy()
    pre_cent_arr= cent_arr_.copy()
    #result = np.where((markers_ == 0)|(cent_arr_ == 1), 0, 255)
    result = np.where((markers_ == 0), 0, 255)


color_makrers = colorizer(result.astype(np.uint8))

cv2.imwrite('pre&diff.jpg', result)
cv2.imwrite('pre&diff_color.jpg', color_makrers)
cv2.imwrite('pre&diff_color_cut.jpg', cutter(a,b,c,d,color_makrers))