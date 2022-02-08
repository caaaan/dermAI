import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import os
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, GlobalAveragePooling2D, Dense, Dropout
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image, label

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')])

root_logdir = os.path.join(os.curdir, "logs\\fit\\")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()

#dataset kısmını değiştirmek/ayarlamak lazım

#(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

train = ImageDataGenerator(rescale= 1/224)
validation = ImageDataGenerator(rescale= 1/224)
CLASS_NAMES= ['eczema','not eczema']



train_dataset = train.flow_from_directory('googlenet/train/', target_size=("224","224"), batch_size=3,class_mode= 'binary') #batch size'ı değiştir
val_dataset = train.flow_from_directory('googlenet/train/', target_size=("224","224"), batch_size=3,class_mode= 'binary') #batch size'ı değiştir



checkpoint = ModelCheckpoint('alexnet.h5', monitor='val_loss', mode='min',save_best_only=True,verbose=1)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0,patience=3,verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,verbose=1,min_delta=0.0001)
callbacks = [earlystop,checkpoint,reduce_lr]


model.compile(loss="binary_crossentropy", optimizer = RMSprop(lr=0.001),metrics = ['accuracy'])

model_fit = model.fit(train_dataset, steps_per_epoch=3, epochs = 10, batch_size=32,callbacks=callbacks,validation_data = val_dataset)
