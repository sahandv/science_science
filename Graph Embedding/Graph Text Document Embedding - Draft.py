#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:46:31 2021

@author: github.com/sahandv
"""
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
# pip install -q tensorflow-datasets
print(tf.__version__)

# =============================================================================
# Settings
# =============================================================================
vocab_size = 100000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = '<OOV_TKN>'

# =============================================================================
# Load tfds data (tensorflow data sets )
# =============================================================================
training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []

imdb,info = tfds.load("imdb_reviews",with_info=True,as_supervised=True)
train_data,test_data = imdb['train'],imdb['test']
for s,l in tqdm(train_data):
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())
for s,l in tqdm(test_data):
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())

testing_labels_final = np.array(testing_labels)
training_labels_final = np.array(training_labels)

# =============================================================================
# Tokenize
# =============================================================================
# sentences = [
#     '1 red brown fox',
#     'red brown cat!',
#     'red cat loves chicken wings.'
#     ]

# oov is out of vocabulary token replacement. you can num_words='int to change vocab size
tokenizer = Tokenizer(oov_token=oov_tok,num_words=vocab_size) 
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
# you can add maxlen=int. you can add padding='post', default is 'pre'. you can truncate='post' to remove from the end, default is 'pre' again
padded_seq = pad_sequences(sequences,maxlen=max_length) 

sequences_test = tokenizer.texts_to_sequences(testing_sentences)
padded_seq_test = pad_sequences(sequences_test,maxlen=max_length) 

# =============================================================================
# Train (see https://keras.io/api/layers/core_layers/embedding/)
# =============================================================================
model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
        tf.keras.layers.Flatten(),
        tf.keras.kayers.Dense(6,activation='relu'),
        tf.keras.kayers.Dense(1,activation='sigmoid'),
     ])

