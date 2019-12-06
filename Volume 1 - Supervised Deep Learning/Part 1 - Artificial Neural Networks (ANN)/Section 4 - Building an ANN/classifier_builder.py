# -*- coding: utf-8 -*-

import keras
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'glorot_uniform'))
    classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'glorot_uniform'))
    classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier