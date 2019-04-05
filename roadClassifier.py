#!/usr/bin/env python

import os
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from pathlib import Path

tf.device('/cpu:0')
img_width, img_height = 150, 150 # shrink the image so it doesnt take forever

train_data_dir = 'PotholeDataset/trainingData'
validation_data_dir = 'PotholeDataset/validationData'

# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(rescale=1./255)

# automagically retrieve images and their classes for train and validation sets
train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=10,
        class_mode='binary')

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=10,
        class_mode='binary')

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5)) # to improve the model so it doesnt just memorize data
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit_generator(
        train_generator,
        samples_per_epoch=3400,
        nb_epoch=2,
        validation_data=validation_generator,
        nb_val_samples=300)

model_structure = model.to_json()
f = Path("PotholeDataset/models/model_structure.json")
f.write_text(model_structure)

model.save_weights("PotholeDataset/models/model_weights.h5")

