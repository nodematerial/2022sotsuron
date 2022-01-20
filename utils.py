import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import math
import seaborn as sns
from logging import Logger, getLogger, INFO, FileHandler,  Formatter,  StreamHandler

def init_logger(log_file='logfile.log'):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter('%(asctime)s: %(message)s   :::%(name)s:%(lineno)s %(funcName)s [%(levelname)s] '))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.propagate = False
    return logger

LOGGER = init_logger()

def colorizer(img):
    result = np.zeros_like(img)[:, :, np.newaxis]
    result = np.concatenate([result,result,result], 2)
    ret, labels = cv2.connectedComponents(img, connectivity=4)
    LOGGER.info(f'=====start color conversion=====')
    for i in tqdm(range(1,ret)):
        if i % 13 == 0:
            result[labels == i] = (50,50,180)
        if i % 13 == 1:
            result[labels == i] = (0,150,0)
        if i % 13 == 2:
            result[labels == i] = (0,150,150)
        if i % 13 == 3:
            result[labels == i] = (120,120,0)
        if i % 13 == 4:
            result[labels == i] = (100,100,140)
        if i % 13 == 5:
            result[labels == i] = (100,140,100)
        if i % 13 == 6:
            result[labels == i] = (140,100,100)
        if i % 13 == 7:
            result[labels == i] = (110,200,200)
        if i % 13 == 8:
            result[labels == i] = (100,100,255)
        if i % 13 == 9:
            result[labels == i] = (173, 255, 47)
        if i % 13 == 10:
            result[labels == i] = (47, 173, 255)
        if i % 13 == 11:
            result[labels == i] = (255, 255, 255)
        if i % 13 == 12:
            result[labels == i] = (55, 55, 25)
    return result

def colorizer2(img):
    result = np.zeros_like(img)[:, :, np.newaxis]
    result = np.concatenate([result,result,result], 2)
    ret = np.max(np.unique(img))
    LOGGER.info(f'=====start color conversion=====')
    for i in tqdm(range(2,ret)):
        if i % 13 == 0:
            result[img == i] = (50,50,180)
        if i % 13 == 1:
            result[img == i] = (0,150,0)
        if i % 13 == 2:
            result[img == i] = (0,150,150)
        if i % 13 == 3:
            result[img == i] = (120,120,0)
        if i % 13 == 4:
            result[img == i] = (100,100,140)
        if i % 13 == 5:
            result[img == i] = (100,140,100)
        if i % 13 == 6:
            result[img == i] = (140,100,100)
        if i % 13 == 7:
            result[img == i] = (110,200,200)
        if i % 13 == 8:
            result[img == i] = (100,100,255)
        if i % 13 == 9:
            result[img == i] = (173, 255, 47)
        if i % 13 == 10:
            result[img == i] = (47, 173, 255)
        if i % 13 == 11:
            result[img == i] = (255, 255, 255)
        if i % 13 == 12:
            result[img == i] = (55, 55, 25)
    return result

def cutter(a,b,c,d,img):
    return img[a:b,c:d]

def opening(img, kernel, blocksize, brightness, iteration, inverse = True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if inverse:
        bin_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blocksize, brightness)
    else:
        bin_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blocksize, brightness)
    kernel = np.ones((kernel, kernel), np.uint8)
    opening = cv2.morphologyEx(
        bin_img, cv2.MORPH_OPEN, kernel, iterations=iteration)
    _, bl_markers, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=4)
    return None, bl_markers, [x[4] for x in stats][1:], centroids

def get_markers(img, kernel, blocksize, brightness, iteration, inverse = True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if inverse:
        bin_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blocksize, brightness)
    else:
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
    ret, markers, stats, _ = cv2.connectedComponentsWithStats(
        sure_fg, connectivity=4)
    markers = markers+1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)    
    markers_ = np.where((markers ==1)|(markers ==0)|(markers ==-1), 0, 255).astype(np.uint8)
    #bl means borderless
    _, bl_markers, stats, centroids = cv2.connectedComponentsWithStats(markers_, connectivity=4)
    return markers, bl_markers, [x[4] for x in stats][1:], centroids

def bin_smallarea_remover(markers,removal):
    markers = markers.copy()
    ret, markers, stats, _ = cv2.connectedComponentsWithStats(
        markers.astype(np.uint8), connectivity=4)    
    for i in range(ret):
        if stats[i, 4] < removal:
            markers = np.where(markers == i, 0, markers)
    markers = np.where(markers==0, 0, 255)
    return markers

def preprocessing(filename, gaussian, cutting = [] ):
    img = cv2.imread(filename)
    top, bottom, left, right = cutting
    assert img is not None, "読み込みに失敗しました"
    if gaussian:
        img = cv2.GaussianBlur(img, (gaussian, gaussian), 0)
    height, width = img.shape[:2]
    img_cut = img[top: height-bottom,
                    left: width-right]
    img_top = img[0: top, left: width-right]
    img_bottom = img[height-bottom: height,
                        left: width-right]
    return (img_cut, img_top, img_bottom)

def border_mask(markers,img):
    img = img.copy()
    img[markers == -1] = [255, 0, 0]
    cv2.imwrite('border_mask.jpg', img)

def border_mask_concat(img, top, bottom, markers):
    img = img.copy()
    img[markers == -1] = [255, 0, 0]
    img = cv2.vconcat([top, img, bottom])
    cv2.imwrite('border_mask_concat.jpg', img)

def border_mask_centroids(img, markers, centroids):
    img = img.copy()
    img[markers == -1] = [255, 0, 0]
    for x, y in centroids[1:]:
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), 2, (0, 0, 255), thickness=-1)
    cv2.imwrite('border_mask_centroids.jpg', img)

def markers_info(size, unit=None, ratio=None):
    LOGGER.info(f'=====markers information=====')
    LOGGER.info(f'領域の個数:{len(size)}')
    LOGGER.info(f'平均サイズ:{np.average(size):.4g} pixel')
    assert bool(unit)^bool(ratio) == False, \
            'unitまたはratioのどちらかのみ指定されています' 
    if unit:
        ave_size = np.average(size)*ratio**2
        LOGGER.info(f'平均サイズ(換算後):{ave_size:.4g} {unit}^2')
        LOGGER.info(f'平均半径(換算後):{np.sqrt(ave_size/np.pi):.4g} {unit}')


def RDF(img, bw, centroids, ratio=1, unit='pixel'):
    if ratio != 1:
        assert unit != 'pixel', 'pixel以外の単位を使用してください'
    img = img.copy()
    distance_list = []
    cent_y, cent_x = img.shape[0]/2, img.shape[1]/2
    minimum = np.inf
    for i, (x, y) in enumerate(centroids[1:]):
        dist = np.linalg.norm([y-cent_y, x-cent_x])
        minimum = min(minimum, dist)
        if minimum == dist:
            min_id = i
    rep_x, rep_y = map(int,centroids[min_id+1])
    radius = min(rep_x, rep_y, img.shape[0]-rep_y, img.shape[1]-rep_x) // 5
    cv2.circle(img, (rep_x,rep_y), 2, (0, 255, 0), thickness=-1)
    for i in range(1,6):
        cv2.circle(img, (rep_x , rep_y), radius*i, (255, 255, 0), thickness=1)
        cv2.putText(img, f'{radius*ratio*i:.5g}'+unit, (img.shape[1]//2, radius*(5+i)), cv2.FONT_HERSHEY_COMPLEX,
            0.5, (0, 0, 0), lineType=cv2.LINE_AA)
    for i, (x, y) in enumerate(centroids[1:]):
        x, y = int(x), int(y)
        if i == min_id:
            pass
        else: 
            cv2.circle(img, (x, y), 2, (0, 0, 255), thickness=-1)
            distance_list.append(np.linalg.norm([y-rep_y, x-rep_x])*ratio)
    distance_list = sorted(list(filter(lambda x: x <= radius*ratio*5, distance_list)))
    #端の影響を無視する
    cut_list = distance_list[:-(int(len(distance_list)*0.1))]
    kde_model = gaussian_kde(distance_list, bw_method=bw)
    split = 1000
    x_grid = np.linspace(0, cut_list[-1],  num=split)
    y = kde_model(x_grid)
    #cut_listの範囲のKDEの積分値が1になるように調整
    y = [i*split/(sum(y)*cut_list[-1]) for i in y]
    density = [y[i]/i for i in range(len(y))]
    #KDE相対度数点画
    fig = plt.figure(figsize=(12,7))
    ax1 = fig.add_subplot(111)
    ax1.plot(x_grid, y)
    ax2 = ax1.twinx()
    ax2.hist(cut_list, alpha=0.3, bins=20)
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel(f"distance ({unit}))")
    ax1.set_ylabel("KDE expected relative frequency")
    ax2.set_ylabel("frequency")
    plt.savefig(f'frequency_{unit}.jpg')
    plt.clf()
    #RDF点画
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(111)
    ax.plot(x_grid, density)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"distance ({unit})")
    ax.set_ylabel("Radial Distribution Function")
    plt.savefig(f'RDF_{unit}.jpg')
    plt.clf()
    cv2.imwrite(f'distance_{unit}.jpg', img)


def size_dist(size, ratio=1, unit='pixel'):
    size = [x*ratio**2 for x in size]
    sns.distplot(size)
    plt.xlabel(f"size ({unit}^2)")
    plt.savefig(f'size_dist_{unit}.jpg')
    plt.clf()

def radius_dist(size, ratio=1, unit='pixel'):
    size = [np.sqrt(x*ratio**2/np.pi) for x in size]
    sns.distplot(size)
    plt.xlabel(f"radius ({unit})")
    plt.savefig(f'radius_dist_{unit}.jpg')
    plt.clf()

def to_circle(H, W, size,centroids):
    LOGGER.info(f'=====start circle conversion=====')
    img = np.zeros([H, W], dtype=np.uint8)
    centroids =centroids[1:].astype(np.int32).tolist()
    radius = [int(np.sqrt(x/np.pi)) for x in size]
    for radius, cent in zip(tqdm(radius),centroids):
        img = draw_circle(cent[0], cent[1], radius, img)
    return img

def draw_circle(x, y, epsilon, img):       
    arr = np.zeros_like(img, dtype=np.uint8)
    for i in range(epsilon):
        for j in range(epsilon):
            if i**2+j**2 < epsilon**2:
                if 0 < y+i < arr.shape[0] and 0 < x+j < arr.shape[1]:
                    arr[y+i][x+j] = 255
                if 0 < y-i < arr.shape[0] and 0 < x+j < arr.shape[1]:
                    arr[y-i][x+j] = 255
                if 0 < y+i < arr.shape[0] and 0 < x-j < arr.shape[1]:
                    arr[y+i][x-j] = 255
                if 0 < y-i < arr.shape[0] and 0 < x-j < arr.shape[1]:
                    arr[y-i][x-j] = 255
    marker = np.where((img==255)|(arr==255),255,0)
    return marker