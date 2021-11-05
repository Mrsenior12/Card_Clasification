
from pylab import *
import skimage
from skimage import data, io, filters, exposure, feature
from skimage.filters import rank
from skimage.util.dtype import convert
from skimage import img_as_float, img_as_ubyte
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
from skimage.filters.edges import convolve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage as ndi
from numpy import array

from skimage.segmentation import watershed


def int2double_digit_int(x: int):
    if (x < 10):
        return '0' + str(x)
    else:
        return str(x)


#def immage_summary(image):
#    image = img_as_float(image)
#    figure(figsize=(20, 20))
#    fig, ax = plt.subplots(nrows=1, ncols=1)
#    matplotlib.pyplot.hist(image, bins=255)
#    xlim(0, 255)
#    plt.show()


def immage_filter(image):
    img2 = filters.sobel(image)
    for i in range(0, img2.shape[0]):
        for j in range(0, img2.shape[1]):
            if (img2[i, j, 0] < 0.09 or img2[i, j, 1] < 0.11 or img2[i, j, 2] < 0.09):
                temp = [0, 0, 0]
                img2[i, j] = tuple(temp)
            if (img2[i, j, 1] - img2[i, j, 0] > 0.07):
                temp = [0, 0, 0]
                img2[i, j] = tuple(temp)
    return img2


#def immage_filter_2(image):
#    img2 = filters.sobel(image)
#    return img2


#def immage_denoise(image):
#    img2 = filters.sobel(image)
#    img2 = img_as_ubyte(img2)
#    return filters.rank.mean(img2, ones([3, 3], dtype=uint8))


#def immage_normalise(image):
#    pass


def immage_stretch_contrast(image):
    img2 = filters.sobel(image)
    MIN = 100 / 256
    MAX = 125 / 256

    norm = (image - MIN) / (MAX - MIN)
    norm[norm > 1] = 1
    norm[norm < 0] = 0
    return norm


def na_3():
    fig, axes = plt.subplots(1, 4, figsize=(20, 20))  # wykresy
    ax = axes.ravel()
    for i in range(0, 4):
        img = img_as_float(io.imread('samoloty/samolot{}.jpg'.format(int2double_digit_int(i))))
        img2 = img
        img2 = immage_filter(img2)
        skimage.io.imsave('na_3/samolot-{}-save.jpg'.format(int2double_digit_int(i)), img2)
            ##wyswietlanie
        ax[i].imshow(img2, cmap=plt.cm.gray)
        ax[i].axis('off')

    fig.tight_layout()
    plt.show()


#def na_5():
#    for i in range(0, 18):
#        img = img_as_float(io.imread('samoloty/samolot{}.jpg'.format(int2double_digit_int(i))))
#        img2 = img
#        img2 = immage_filter(img2)
#        skimage.io.imsave('na_5/samolot-{}-save.jpg'.format(int2double_digit_int(i)), img2)

if __name__ == '__main__':
    na_3()