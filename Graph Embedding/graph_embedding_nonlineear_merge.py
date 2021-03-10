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
import tensorflow as tf
from tensorflow.contrib.keras import layers

# read labels

# read doc vecs

# read graph vecs

# split data based on the labels and make them more balanced

# split data for train/validation/test

# train

# get output 
