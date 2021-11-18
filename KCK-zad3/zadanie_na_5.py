from pylab import *
import skimage
from skimage import data, io, filters, exposure, feature,measure,morphology,img_as_float, img_as_ubyte
from skimage.filters import rank
from skimage.util.dtype import convert
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import find_contours, approximate_polygon,subdivide_polygon
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage as ndi
from numpy import array

def load_img(img):
    dt = io.imread(img)
    dt = (img_as_float(dt))
    return dt

def double_digit_int(x: int):
    if (x < 10):
        return '0' + str(x)
    else:
        return str(x)

def immage_stretch_contrast(image,perc=5):
    MIN = np.percentile(image,perc)
    MAX = np.percentile(image,100-perc)
    norm = (image - MIN) / (MAX - MIN)
    norm[norm > 1] = 1.0
    norm[norm < 0] = 0.0
    return norm

filelist = ['samoloty/samolot00.jpg','samoloty/samolot01.jpg','samoloty/samolot02.jpg','samoloty/samolot03.jpg','samoloty/samolot04.jpg','samoloty/samolot05.jpg','samoloty/samolot06.jpg','samoloty/samolot00.jpg','samoloty/samolot00.jpg','samoloty/samolot07.jpg','samoloty/samolot08.jpg','samoloty/samolot09.jpg','samoloty/samolot10.jpg','samoloty/samolot11.jpg','samoloty/samolot12.jpg','samoloty/samolot13.jpg','samoloty/samolot14.jpg','samoloty/samolot15.jpg','samoloty/samolot16.jpg','samoloty/samolot17.jpg','samoloty/samolot18.jpg','samoloty/samolot19.jpg','samoloty/samolot20.jpg']
if __name__ == '__main__':
    for i in range(0,len(filelist)):
        figure(figsize=(100,100))
        subplot(21,1,i+1)
        image = load_img(filelist[i])
        img = immage_stretch_contrast(image,0.25)
        black = morphology.erosion(morphology.dilation(filters.gaussian(filters.sobel(rgb2gray(img)))))
        black[black < 0.06] = 0
        black[black > 0.06] = 1
        black = ndi.binary_fill_holes(black)
        contoures = measure.find_contours(black,0,fully_connected='high',positive_orientation='high')
        for j,conter in enumerate(contoures):
            if len(conter) > 250 and conter[0][0] == conter[-1][0] and conter[0][1] == conter [-1][1]:
                test = approximate_polygon(conter,tolerance=3.5)
                plt.scatter(mean(test[:,1]),mean(test[:,0]),color='y',s=10)
                plt.plot(test[:,1],test[:,0],linewidth=2)
        plt.imshow(image)
        plt.show()
