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
    prevHist = []
    X_train.append(training_set_scaled[i-60:i])
    y_train.append(training_set_scaled[i])

""" Part 2 - Building the RNN """


""" Part 3 - Make predictions and visualize the results """