import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import pandas as pd
import cv2
import skimage
from skimage import io
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

def create_model(log,train_X,train_Y,test_X,test_Y):
    model = tf.keras.models.Sequential()
    if log.at[0,'model'] == 0:
        # Adding Specification to our model.
        model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(train_X.shape[1],train_X.shape[2],train_X.shape[3])))
        model.add(tf.keras.layers.MaxPooling2D(2,2))
        model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(2,2))
        model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(2,2))
        model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(2,2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(512,activation='relu'))
        model.add(tf.keras.layers.Dense(52,activation='softmax')) 

        # Training our new model.
        cp = tf.keras.callbacks.ModelCheckpoint(filepath='150epochs.h5',save_best_only=True,verbose=0)
        model.compile(loss = 'sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.fit(train_X,train_Y,epochs=2,validation_data=(test_X,test_Y),callbacks=[cp])

        model.save('model.h5')
        log.at[0,'model'] = 1
    else:
        model = tf.keras.models.load_model('model.h5')

    return model


    
    #musimy zwrócic model który został uczony 

log = pd.read_csv("info.csv")
log = log.replace(np.nan,0).astype(np.int64) #zaczytujemy plik CSV w którym są puste wartości dla kolumn, po czym zastępujemy je 0

data_gen = ImageDataGenerator(rotation_range=90,brightness_range=(0.5,1.5),shear_range=15.0,zoom_range=[.3,.8])

#pokaz_przyklad(data_gen)

#gather data which will be used for training and testing our model.
data = modify_cards(log,data_gen)
training_data = data[:4800]
training_X = np.array([x[0] for x in training_data])
training_Y = np.array([x[1] for x in training_data])

test_data = data[4800:]
test_X = np.array([x[0] for x in test_data])
test_Y = np.array([x[1] for x in test_data])

model = create_model(log,training_X,training_Y,test_X,test_Y)
model.summary()
log.to_csv("info.csv",index=False)

def show_cards(test_X, model):
    labs = ['2 of clubs', '2 of diamonds', '2 of spades', '2 of hearts',
        '3 of clubs', '3 of diamonds', '3 of spades', '3 of hearts',
        '4 of clubs', '4 of diamonds', '4 of spades', '4 of hearts',
        '5 of clubs', '5 of diamonds', '5 of spades', '5 of hearts',
        '6 of clubs', '6 of diamonds', '6 of spades', '6 of hearts',
        '7 of clubs', '7 of diamonds', '7 of spades', '7 of hearts',
        '8 of clubs', '8 of diamonds', '8 of spades', '8 of hearts',
        '9 of clubs', '9 of diamonds', '9 of spades', '9 of hearts',
        '10 of clubs', '10 of diamonds', '10 of spades', '10 of hearts',
        'jack of clubs', 'jack of diamonds', 'jack of spades', 'jack of hearts',
        'queen of clubs','queen of diamonds','queen of spades','queen of hearts',
        'king of clubs','king of diamonds','king of spades','king of hearts',
        'as of clubs','as of diamonds','as of spades','as of hearts']
    predict_x=model.predict(test_X) 
    predictions=np.argmax(predict_x,axis=1)

    sample=test_X[:20]
    plt.figure(figsize=(10,10))
    for i in range(20):
        plt.subplot(5,4,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(sample[i].reshape(sample.shape[1], sample.shape[2]))
        plt.xlabel(labs[predictions[i]])
    plt.show()

show_cards(test_X, model)



