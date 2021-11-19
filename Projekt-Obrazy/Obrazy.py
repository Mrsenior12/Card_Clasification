import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import pandas as pd
import cv2
import skimage
from skimage import io
import os
from random import shuffle
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

def pokaz_przyklad(generator):
    img = plt.imread('karty/02_1.JPG')
    imgs = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))

    plt.imshow(imgs[0])
    plt.show()

    generator.fit(imgs)
    image_iter = generator.flow(imgs)
    plt.figure(figsize=(4,4))
    for i in tqdm(range(4)):
        plt.subplot(1,4,i+1)
        plt.imshow(image_iter.next()[0].astype('int'))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
    plt.show()


def mod_karty(log,generator):
    #tutaj zajdzie utworzenie 4 różnych zdjęć do 1 karty
    if log.at[0,'modyfikacja_kart'] == 0:
        for i, images in tqdm(enumerate(os.listdir('karty/'))):#Pętla zaczytująca zdjęcia z folderu karty

            name,ext = images.split('.') #wydobycie nazwy zdjęcia
            img = plt.imread('karty/{}'.format(images)) #zaczytanie zdjęcia 
            imgs = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
            print(img.shape)

            for j in range(1,5): #pętla w której przekształcamy jedno zdjęcie w 4 zmodyfikowane zdjęcia przy użyciu ImageDataGenerator
                generator.fit(imgs)
                image_iter = generator.flow(imgs)
                skimage.io.imsave('kartyP/{}_{}.jpg'.format(name,j),image_iter.next()[0].astype('int'))  

        log.at[0,'modyfikacja_kart'] = 1 # zapisanie informacji o tym czy już zmodyfikowaliśmy karty.


log = pd.read_csv("info.csv")
log = log.replace(np.nan,0).astype(np.int64) #zaczytujemy plik CSV w którym są puste wartości dla kolumn, po czym zastępujemy je 0

data_gen = ImageDataGenerator(rotation_range=90,brightness_range=(0.5,1.5),shear_range=15.0,zoom_range=[0.3,.8])

pokaz_przyklad(data_gen)
mod_karty(log,data_gen)

log.to_csv("info.csv",index=False)
