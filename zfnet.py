import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, GlobalAveragePooling2D, Dense, Dropout
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



model = tf.keras.models.Sequential([
                                    
		tf.keras.layers.Conv2D(96, (7, 7), strides=(2, 2), activation='relu',
			input_shape=(224, 224, 3)),
		tf.keras.layers.MaxPooling2D(3, strides=2),
    tf.keras.layers.Lambda(lambda x: tf.image.per_image_standardization(x)),

		tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), activation='relu'),
		tf.keras.layers.MaxPooling2D(3, strides=2),
    tf.keras.layers.Lambda(lambda x: tf.image.per_image_standardization(x)),

		tf.keras.layers.Conv2D(384, (3, 3), activation='relu'),

		tf.keras.layers.Conv2D(384, (3, 3), activation='relu'),

		tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),

		tf.keras.layers.MaxPooling2D(3, strides=2),

    tf.keras.layers.Flatten(),

		tf.keras.layers.Dense(4096),

		tf.keras.layers.Dense(4096),

		tf.keras.layers.Dense(10, activation='softmax')
	])


#model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9),loss='categorical_crossentropy',
#metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(5)])
#reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=1,min_lr=0.00001)

#model.fit(training_images, training_labels, batch_size=128,validation_data=(valid_images, valid_labels),epochs=90, callbacks=[reduce_lr])


train = ImageDataGenerator(rescale= 1/224)
validation = ImageDataGenerator(rescale= 1/224)

train_dataset = train.flow_from_directory('googlenet/train/', 
                target_size=("224","224"), batch_size=3,class_mode= 'binary') #batch size'ı değiştir
val_dataset = train.flow_from_directory('googlenet/train/', 
                target_size=("224","224"), batch_size=3,class_mode= 'binary') #batch size'ı değiştir

checkpoint = ModelCheckpoint('zfnet.h5', monitor='val_loss', mode='min',save_best_only=True,verbose=1)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0,patience=3,verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,verbose=1,min_delta=0.0001)
callbacks = [earlystop,checkpoint,reduce_lr]


model.compile(loss="binary_crossentropy", optimizer = RMSprop(lr=0.001),metrics = ['accuracy'])

model_fit = model.fit(train_dataset, steps_per_epoch=3, epochs = 10, batch_size=32,callbacks=callbacks,validation_data = val_dataset)
