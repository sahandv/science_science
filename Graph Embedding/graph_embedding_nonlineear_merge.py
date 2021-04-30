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
from keras.utils.vis_utils import plot_model
from tensorflow.keras.layers import BatchNormalization

from keras_visualizer import visualizer 
from tensorflow.keras.callbacks import TensorBoard

pretrain = True
get_output = True

# read labels
datapath = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/'
# datapath = '/home/sahand/GoogleDrive/Data/'
data_dir =  datapath+'Corpus/cora-classify/cora/'
label_address =  data_dir+'clean/single_component_small_18k/labels'
labels = pd.read_csv(label_address)
num_classes = len(labels.groupby('class1').groups.keys())
population_classes =  [len(dict(labels.groupby('class1').groups)[x]) for x in dict(labels.groupby('class1').groups)]
# read doc vecs
vec_a = pd.read_csv(data_dir+'embeddings/single_component_small_18k/doc2vec 300D dm=1 window=10')

# read graph vecs
vec_b = pd.read_csv(data_dir+'embeddings/single_component_small_18k/cocite node2vec 300-70-20 p1q05',index_col=0)
# vec_c = pd.read_csv(data_dir+'embeddings/single_component_small_18k/n2v 300-70-20 p1q05',index_col=0).reset_index().drop(['index'],axis=1)

# concat vecs
vec = pd.concat([vec_a,vec_b],axis=1)
vec.columns = [i for i in range(vec.shape[1])]
vec = vec.reset_index()
# split data based on the labels 
# y_train = tf.keras.utils.to_categorical(labels['class1'].values, num_classes=num_classes)
Y = pd.get_dummies(labels).values
X = vec.values
# make them more balanced

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.5, random_state=100,shuffle=True)
test_idx = pd.DataFrame(Xtest[:,0],columns=['id_seq'])
test_idx.to_csv(data_dir+'clean/single_component_small_18k/nonlinear_test_data_idx 20 april',index=False)
Xtrain = np.delete(Xtrain,0,1) 
X = np.delete(X,0,1) 
del Xtest

gc.collect()

if pretrain:
    # split data for train and future test
    Xpretrain_train, Xpretrain_test, Ypretrain_train, Ypretrain_test = train_test_split(Xtrain, Ytrain, test_size=0.3, random_state=100,shuffle=True)
    
    
    # define model
    model_shape=[800,600,200,100]
    input_shape = X.shape[1]#600
    acts=['relu','relu','relu','sigmoid']
    act_last='softmax'
    loss_f='categorical_crossentropy'
    opt='adam'
    init='glorot_uniform'
    
    x = Input(shape=(input_shape,), name='input')
    x_1 = x
    x_1 = BatchNormalization(name='b_normalization_'+str(0))(x_1)
    for i,dim in enumerate(model_shape[:-1]):
        x_1 = Dense(dim, activation=acts[i], name='hidden_'+str(i))(x_1)        
    
    x_2 = Dense(model_shape[-1], activation=acts[i+1], name='hidden_'+str(i+1))(x_1)
    # x_3 = Dense(model_shape[3], activation=act, kernel_initializer=init, name='hidden_3')(x_2)
    
    x_2 = concatenate([x_1, x_2], name='embedding')
    
    y = Dense(num_classes, activation=act_last, name='output')(x_2)
    model = Model(inputs=x, outputs=y, name='classifier')
    model.compile(loss=loss_f, optimizer=opt, metrics=['accuracy'])
    model.summary()
    
    callback = EarlyStopping(monitor='val_accuracy',patience=35)
    checkpoint = ModelCheckpoint('models/pretrain_best_model-cocite_.h5', monitor='val_accuracy', mode='min', save_best_only=True)
    tensorboard = TensorBoard(
                              log_dir='.\logs',
                              histogram_freq=1,
                              write_images=True
                            )
    print('classes',dict(pd.DataFrame(pd.DataFrame(Ypretrain_train).idxmax(axis=1),columns=['label']).groupby('label').groups).keys())
    # Xpretrain_test.shape
    # train
    history = model.fit(Xpretrain_train, Ypretrain_train, validation_data=(Xpretrain_test, Ypretrain_test), epochs=400, batch_size = 64, verbose=1, callbacks=[callback, checkpoint,tensorboard])

model.summary()

# get output 
if get_output:
    input_arr = X
    
    best_model = load_model('models/pretrain_best_model-cocite_.h5')
    plot_model(best_model, to_file='models/model_plot.png', show_shapes=True, show_layer_names=True)
    
    # get_output = K.function([best_model.input],[best_model.layers[5].output])
    # output = get_output([np.expand_dims(input_arr,axis=0)])[0]
    # output = get_output([input_arr])[0]
    
    layer_name = 'embedding'
    # layer_name = 'hidden_2'
    intermediate_layer_model = Model(inputs=best_model.input,outputs=best_model.get_layer(layer_name).output)
    embeddings = pd.DataFrame(intermediate_layer_model.predict(X))
    embeddings.to_csv(datapath+'Corpus/cora-classify/cora/embeddings/single_component_small_18k/deep_nonlinear_embedding_600_50percent 20 April',index=False)
    
    
    