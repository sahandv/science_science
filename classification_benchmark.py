#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:10:03 2020

@author: sahand
"""
import sys
import time
import gc
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import pyplot
from random import randint

from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras import backend as K
import keras as keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint

from sciosci.assets import text_assets as ta
from DEC.DEC_keras import DEC_simple_run
np.random.seed(100)
# =============================================================================
# Evaluation method
# =============================================================================
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
# =============================================================================
# Classify and evaluate
# =============================================================================
def run_all_tests(data_address:str,output_dir:str,labels:list,model_shapes:list,test_size:float=0.1):
    tic = time.time()
    vectors = pd.read_csv(data_address)#,header=None,)
    try:
        vectors = vectors.drop('Unnamed: 0',axis=1)
        print('\nDroped index column. Now '+data_address+' has the shape of: ',vectors.shape)
    except:
        print('\nVector shapes seem to be good:',vectors.shape)
        
    path_to_model = output_dir+'classification/'+data_address.split('/')[-1]
    Path(path_to_model).mkdir(parents=True, exist_ok=True)
    
    # preprocess inputs and split
    labels_f = pd.factorize(labels.label)[0]
    enc = OneHotEncoder(handle_unknown='ignore')
    Y = enc.fit_transform(np.expand_dims(labels_f,axis=1)).toarray()
    X = vectors.values
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=test_size, random_state=100,shuffle=True)
    Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(Xtrain, Ytrain, test_size=0.3, random_state=100,shuffle=True)
    
    results = []
    
    for i,model_shape in enumerate(model_shapes):
        
        #run models
        model = Sequential()
        
        model.add(Dense(model_shape[0], input_dim=128, activation='relu', kernel_initializer='he_uniform'))
        for dim in model_shape[1:-1]:
            model.add(Dense(dim, activation='relu'))
        model.add(Dense(model_shape[-1], activation='softmax'))
        # compile model
        # opt = SGD(lr=0.01, momentum=0.9)
        opt = Adam(learning_rate=0.1)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])  #sparse_categorical_crossentropy  if you don't want to categorize yourself
        print(model.summary())
        
        # fit model
        path_to_model = path_to_model + '/' + str(model_shape) + '_checkpoint.hdf5'
        print('The model will be saved into ',path_to_model)
        checkpoint = ModelCheckpoint(filepath=path_to_model, save_best_only=True, monitor='val_accuracy', mode='max')
        
        history = model.fit(Xtrain, Ytrain, validation_data=(Xvalid, Yvalid), epochs=100, verbose=0, callbacks=[checkpoint])
        
        #test model
        # loss, accuracy, f1_score, precision, recall
        results.append(model.evaluate(Xtest, Ytest, verbose=0))
        
        _, train_acc = model.evaluate(Xtrain, Ytrain, verbose=0)
        _, test_acc = model.evaluate(Xtest, Ytest, verbose=0)
        
        print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
        
        # plot loss during training
        pyplot.subplot(211)
        pyplot.title('Loss')
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        # plot accuracy during training
        pyplot.subplot(212)
        pyplot.title('Accuracy')
        pyplot.plot(history.history['accuracy'], label='train')
        pyplot.plot(history.history['val_accuracy'], label='test')
        pyplot.legend()
        pyplot.title(str(model_shape))
        pyplot.show()
    
    column_names = ['loss','accuracy','f1','percision','recall']
    results = pd.DataFrame(results,columns=column_names)
    toc = time.time()
    print('All done in '+str(toc - tic)+'seconds!')
    return results
# =============================================================================
# Run
# =============================================================================
datapath = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/'
data_dir =  datapath+"Corpus/cora-classify/cora/"
label_address =  datapath+"Corpus/cora-classify/cora/clean/with citations new/corpus classes1"

# vec_file_names = ['embeddings/node2vec super-d2v-node 128-80-10 p4q1','embeddings/node2vec super-d2v-node 128-80-10 p1q025','embeddings/node2vec super-d2v-node 128-10-100 p1q025']#,'Doc2Vec patent corpus',
                  # ,'embeddings/node2vec-80-10-128 p1q0.5','embeddings/node2vec deepwalk 80-10-128']
# vec_file_names =  ['embeddings/node2vec super-d2v-node 128-80-10 p1q05']
vec_file_names =  ['embeddings/node2vec super-d2v-node 128-10-100 p1q025']

labels = pd.read_csv(label_address)
labels.columns = ['label']

model_shapes = [
    [512,256,64,10],
    [1024,256,16,10],
    [512,64,10]
    ]
results = []
for file_name in vec_file_names:
    gc.collect()
    data_address = data_dir+file_name
    output_dir = data_dir
    results.append(run_all_tests(data_address,data_dir,labels,model_shapes))

