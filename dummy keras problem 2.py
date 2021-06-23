#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:30:51 2021

@author: sahand
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
#from keras.utils import np_utils
#from sklearn.preprocessing import LabelEncoder
seed = 10
np.random.seed(seed)
# Import data
df = pd.read_csv('data/Sensorless_drive_diagnosis.txt', sep = ' ', header = None)
# Print first 10 samples
print(df.head(10))
print(df.isna().sum())

X = df.loc[:,0:47]
Y = df.loc[:,48]
print(X.shape)
print(Y.shape)

print(X.describe())

print(df.groupby(Y).size())

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
X = pd.DataFrame(X)

Y = pd.get_dummies(Y)

X = X.values
Y = Y.values

def baseline_model():
    # Create model here
    model = Sequential()
    model.add(Dense(15, input_dim = 48, activation = 'relu')) # Rectified Linear Unit Activation Function
    model.add(Dense(15, activation = 'relu'))
    model.add(Dense(11, activation = 'softmax')) # Softmax for multi-class classification
    # Compile model here
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

estimator = KerasClassifier(build_fn = baseline_model, epochs = 100, batch_size = 10, verbose = 0)
kfold = KFold(n_splits = 5, shuffle = True, random_state = seed)
results = cross_val_score(estimator, X, Y, cv = kfold)
print("Result: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))






