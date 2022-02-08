import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras 
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(strides=2))
model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPool2D(strides=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

train = ImageDataGenerator(rescale= 1/224)
validation = ImageDataGenerator(rescale= 1/224)

train_dataset = train.flow_from_directory('googlenet/train/', 
                target_size=("224","224"), batch_size=3,class_mode= 'binary') #batch size'ı değiştir
val_dataset = train.flow_from_directory('googlenet/train/', 
                target_size=("224","224"), batch_size=3,class_mode= 'binary') #batch size'ı değiştir


checkpoint = ModelCheckpoint('lenet.h5', monitor='val_loss', mode='min',save_best_only=True,verbose=1)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0,patience=3,verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,verbose=1,min_delta=0.0001)
callbacks = [earlystop,checkpoint,reduce_lr]


model.compile(loss="binary_crossentropy", optimizer = RMSprop(lr=0.001),metrics = ['accuracy'])

model_fit = model.fit(train_dataset, steps_per_epoch=3, epochs = 10, batch_size=32,callbacks=callbacks,validation_data = val_dataset)
