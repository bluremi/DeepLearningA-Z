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

""" This just verifies that GPU is being used by tensorflow"""
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


""" Part 2 - Making the Artifical Neural Network """

# Import kera libraries and other required packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialize the ANN by defining it as a sequence of layers
classifier = Sequential()
# Two hidden layers. Rule of thumb for hidden nodes = average of input + outputs, or 11+1/2 = 6
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'glorot_uniform'))
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'glorot_uniform'))

# Output layer
classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))

# Compile the ANN. Apply stochastic gradient descdent on the whole ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit the ANN to the training set
classifier.fit(x = X_train, y = y_train, batch_size = 10, epochs = 100)



"""  Part 3 - Generate predictions and evaluate the model """

# Predict the Test set results
y_predictions = classifier.predict(X_test)

# Apply a threshold function (0.5) to convert the probabilities in the prediction set into binary results
y_predictions = (y_predictions > 0.5)

# Generate a Confusion Matrix to evaluate the accuracy of the predictions
from sklearn.metrics import confusion_matrix
cmatrix = confusion_matrix(y_test, y_predictions)




""" Homework
Use our ANN model to predict if the customer with the following informations will leave the bank: 

Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
"""

# Prepare the data by transforming using the fitted transformers from earlier steps
hw_data = pd.read_csv('Churn_Modelling_Homework.csv')
hw_inputs = hw_data.iloc[10000:, 3:13].values
hw_inputs[:, 1] = labelencoder_X_1.transform(hw_inputs[:, 1])
hw_inputs[:, 2] = labelencoder_X_2.transform(hw_inputs[:, 2])
hw_inputs = onehotencoder.transform(hw_inputs).toarray()
hw_inputs = hw_inputs[:, 1:]
hw_inputs = sc.transform(hw_inputs) #feature scaling

# Could also generate the row manually via hw_data = np.array([[France, 0, 1, 600, 2, etc...]])

# Generate a prediction on the 1 record
hw_prediction = classifier.predict(hw_inputs)



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
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 2)


