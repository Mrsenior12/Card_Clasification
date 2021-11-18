import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import pandas as pd
import cv2
import os
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def pokaz_przyklad():
    img = plt.imread('karty/12-1.jpg')
    imgs = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
    plt.imshow(imgs[0])
    plt.show()

    data_gen = ImageDataGenerator(rotation_range=90,brightness_range=(0.5,1.5),shear_range=15.0,zoom_range=[0.3,1.0])
    data_gen.fit(imgs)
    image_iter = data_gen.flow(imgs)

    plt.figure(figsize=(4,4))
    for i in tqdm(range(4)):
        plt.subplot(1,4,i+1)
        plt.imshow(image_iter.next()[0].astype('int'))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
    plt.show()


def mod_karty(log):
    if log.at[0,'modyfikacja_kart'] == 0:
        print("tak")

        log.at[0,'modyfikacja_kart'] = 1
    else:
        print("niee")
    
#    return log

log = pd.read_csv("info.csv")
log = log.replace(np.nan,0).astype(np.int64) #zaczytujemy plik CSV w którym są puste wartości dla kolumn, po czym zastępujemy je 0
mod_karty(log)
pokaz_przyklad()
#for images in tqdm(enumerate(os.listdir('karty/'))):
#mod_karty(log)
#print(log)
log.to_csv("info.csv",index=False)
