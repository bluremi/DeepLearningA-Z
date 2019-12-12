# -*- coding: utf-8 -*-
"""
Recurrent Neural Networks - stock price predictor

@author: Phil
"""

""" Part 1 - Data Pre-processing """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Create a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60,training_set_scaled.size - 1):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Add a dimension to the numpy array using Reshaping function
# The new dimension indexes the model inputs. In our model we only have a single input (open price) so there's just one dimension
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

""" Part 2 - Building the RNN """
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

# Add four LSTM layers with Dropout regularization to avoid over-fitting
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(rate = 0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(rate = 0.2))



""" Part 3 - Make predictions and visualize the results """