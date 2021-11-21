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

def show_example(gen):
    img = plt.imread('karty/02_1.JPG')
    imgs = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))

    plt.imshow(imgs[0])
    plt.show()

    gen.fit(imgs)
    image_iter = gen.flow(imgs)
    plt.figure(figsize=(4,4))
    for i in tqdm(range(4)):
        plt.subplot(1,4,i+1)
        plt.imshow(image_iter.next()[0].astype('int'))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
    plt.show()


def modify_cards(log,gen):
    data = []
    # loop over directory with cards and modify each 100 times as shown in function SHOW_EXAMPLE
    # afther that save that data so we don't have to repeat that procces every single time
    if log.at[0,'przygotowane'] == 0:
        for i,images in tqdm(enumerate(os.listdir('karty/'))): 
            img = cv2.imread('karty/{}'.format(images),cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(180,180))
            imgs = img.reshape((1,img.shape[0],img.shape[1],1))
            gen.fit(imgs)
            image_iter = gen.flow(imgs)
            for j in range(100):
                img_transformed = image_iter.next()[0].astype('int')/255
                data.append([img_transformed,i])
        shuffle(data)
        np.save('data.npy',data)
        log.at[0,'przygotowane'] = 1
    else:
        data = np.load('data.npy',allow_pickle=True)
    
    return data

#def przygotuj_dane(dane):

def nauczanie(log):
    if log.at[0,'trening_sieci'] == 0:
        print("tutaj jest trening sieci")
    else:
        print("zaczytanie pliku z uczeniem")
    
    #musimy zwrócic model który został uczony 

log = pd.read_csv("info.csv")
log = log.replace(np.nan,0).astype(np.int64) #zaczytujemy plik CSV w którym są puste wartości dla kolumn, po czym zastępujemy je 0

data_gen = ImageDataGenerator(rotation_range=90,brightness_range=(0.5,1.5),shear_range=15.0,zoom_range=[0.3,.8])

#pokaz_przyklad(data_gen)
data = modify_cards(log,data_gen)
print(len(data))
log.to_csv("info.csv",index=False)
