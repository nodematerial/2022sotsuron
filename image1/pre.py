import cv2
import os
import sys
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.abspath(".."))
from utils import colorizer, cutter

gaussian = 5
brightness = 8
kernel = 3
iteration = 3
blocksize = 41
a = 100
b = 500
c = 1000
d = 1400

img = cv2.imread('image1.jpg')
img = cv2.GaussianBlur(img, (gaussian, gaussian), 0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bin_img = cv2.adaptiveThreshold(
gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blocksize, brightness)

kernel = np.ones((kernel, kernel), np.uint8)
opening = cv2.morphologyEx(
    bin_img, cv2.MORPH_OPEN, kernel, iterations=iteration)

color_makrers = colorizer(opening)

cv2.imwrite('pre.jpg', opening)
cv2.imwrite('pre_color.jpg', color_makrers)
cv2.imwrite('pre_color_cut.jpg', cutter(a,b,c,d,color_makrers))