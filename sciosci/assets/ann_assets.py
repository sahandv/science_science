#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 12:31:43 2021

@author: github.com/sahandv
"""
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_graphs(history, string="accuracy"):
    """
    Draw from history
    
    Parameters
    ----------
    history : TensorFlow fit history
    string : string
        Draw objective. Default is accuracy. 
    
    Returns
    -------
    None.
    
    """
    
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()







class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df, X_col, y_col,
                 batch_size=64,
                 shuffle=True):
        
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.n = len(self.df)
        self.n_name = df[y_col['name']].nunique()
        self.n_type = df[y_col['type']].nunique()
    
    def on_epoch_end(self):
        pass
    
    def __getitem__(self, index):
        pass
    
    def __len__(self):
        return self.n // self.batch_size







