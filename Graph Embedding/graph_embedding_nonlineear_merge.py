#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:50:38 2021

@author: github.com/sahandv

take ideas from:
    https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
    https://github.com/susanli2016/NLP-with-Python/blob/master/Multi-Class%20Text%20Classification%20LSTM%20Consumer%20complaints.ipynb
    https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
"""

import sys
import gc
import pandas as pd
import numpy as np
import networkx as nx
import karateclub as kc
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras as keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input, Dropout, concatenate
from keras.models import Sequential, load_model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.np_utils import to_categorical# from tensorflow.contrib.keras import layers

# read labels
# datapath = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/'
datapath = '/home/sahand/GoogleDrive/Data/'
data_dir =  datapath+'Corpus/cora-classify/cora/'
label_address =  data_dir+'clean/single_component_small/labels'
labels = pd.read_csv(label_address)
num_classes = len(labels.groupby('class1').groups.keys())
population_classes =  [len(dict(labels.groupby('class1').groups)[x]) for x in dict(labels.groupby('class1').groups)]
# read doc vecs
vec_a = pd.read_csv(data_dir+'embeddings/single_component_small/doc2vec 300D dm=1 window=10')

# read graph vecs
vec_b = pd.read_csv(data_dir+'embeddings/single_component_small/node2vec 300-70-20 p1q05',index_col=0)

# concat vecs
vec = pd.concat([vec_a,vec_b],axis=1)
vec.columns = [i for i in range(vec.shape[1])]

# split data based on the labels 
# y_train = tf.keras.utils.to_categorical(labels['class1'].values, num_classes=num_classes)
Y = pd.get_dummies(labels).values
X = vec.values
# make them more balanced


# split data for train and future test
Xpretrain, Xtrain, Ypretrain, Ytrain = train_test_split(X, Y, test_size=0.6, random_state=100,shuffle=True)
Xpretrain_train, Xpretrain_test, Ypretrain_train, Ypretrain_test = train_test_split(X, Y, test_size=0.3, random_state=100,shuffle=True)


# define model
model_shape=[800,300,100]
input_shape = 600
act='relu'
act_last='softmax'
loss_f='categorical_crossentropy'
opt='adam'
init='glorot_uniform'

x = Input(shape=(input_shape,), name='input')
x_1 = x
for i,dim in enumerate(model_shape[:-1]):
    x_1 = Dense(dim, activation=act, name='hidden_'+str(i))(x_1)

x_2 = Dense(model_shape[-1], activation=act, name='hidden_last')(x_1)
# x_3 = Dense(model_shape[3], activation=act, kernel_initializer=init, name='hidden_3')(x_2)

x_2 = concatenate([x_1, x_2])
y = Dense(num_classes, activation=act_last, name='output')(x_2)#(combined)
model = Model(inputs=x, outputs=y, name='classifier')
model.compile(loss=loss_f, optimizer=opt, metrics=['accuracy'])
model.summary()

callback = EarlyStopping(monitor='val_loss',patience=80)
checkpoint = ModelCheckpoint('models/pretrain_best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

print('classes',dict(pd.DataFrame(pd.DataFrame(Ypretrain_train).idxmax(axis=1),columns=['label']).groupby('label').groups).keys())
# Xpretrain_test.shape
# train
history = model.fit(Xpretrain_train, Ypretrain_train, validation_data=(Xpretrain_test, Ypretrain_test), epochs=400, batch_size = 32, verbose=1, callbacks=[callback, checkpoint])

# get output 
# best_model = load_model('models/pretrain_best_model.h5')


