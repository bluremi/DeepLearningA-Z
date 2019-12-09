#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convolutional Neural Network

@author: phil.brosgol
"""

"""Part 1 - Building the Convolutional Neural Network """
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# initialize the CNN
classifier = Sequential()
#Step 1 - Convolution
# NOTE: When using tensorflow backend, the order for input_shape is row/col/channels, but for theano it's reversed
classifier.add(Convolution2D(nb_filter = 32, nb_row = 3, nb_col = 3, 
                             input_shape = (64, 64, 3), #X, Y, channels (RGB)
                             activation = 'relu'))
#Step 2 - max pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#Step 3 - flatten layers
classifier.add(Flatten())
#Step 4 - full connection (implement the classic ANN)
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
# If we had multiple categories instead of binary, we would use categorical_crossentropy
classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])


"""Part 2 - Fitting the CCN to the images 
With images, it is very easy to overfit to the training set, due to variance in data
and a low volume of images
We are going to use a keras shortcut to create some modified images from our 8000 original
data set in order to augment the size of the training data set"""
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64,64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(generator = training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)















