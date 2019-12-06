#Artifial Neural Network

"""
Created on Tue Nov 26 14:29:13 2019

@author: phil.brosgol
"""

""" Part 1 - Data Preprocessing """

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""SOME TEST STUFF TO FIX THE TENSORFLOW PROBLEMS """
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
#tf.config.experimental.set_visible_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



# Import the actual dataset
dataset= pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encode the categorical data columns into numerical data types
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encode country of origin into an integer column
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# Encode gender into a binary column
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# Convert country column into dummy variable columns and then remove one of them to avoid dummy variable trap
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


#split dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# We call transform instead of fit_transform because we already fit the scaler to the training dataset
X_test = sc.transform(X_test)




""" Part 2 - Making the Artifical Neural Network """

# Import kera libraries and other required packages
import keras
from keras.models import Sequential
from keras.layers import Dense

""" Part 4 - Evaluating and improving and tuning the ANN """

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'glorot_uniform'))
    classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'glorot_uniform'))
    classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


# create a classifier that's wrapped in the KerasClassifier so that we can use the cross_val_score function to evaluate
# The first argument of the function is itself a function, which returns a classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 10)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)


