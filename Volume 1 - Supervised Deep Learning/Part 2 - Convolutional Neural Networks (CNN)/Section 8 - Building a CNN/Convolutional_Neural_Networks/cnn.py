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
                             input_shape = (64, 64, 3), 
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
                   metrics = )


