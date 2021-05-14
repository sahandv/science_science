#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:46:31 2021

@author: github.com/sahandv
"""
import io
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# pip install -q tensorflow-datasets
print(tf.__version__)

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
# =============================================================================
# Settings
# =============================================================================
vocab_size = 100000
embedding_dim = 128
max_length = 120
trunc_type = 'post'
oov_tok = '<OOV_TKN>'
num_epochs = 50

# =============================================================================
# Load tfds data (tensorflow data sets )
# =============================================================================
training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []

imdb,info = tfds.load("imdb_reviews",data_dir='data/tfds',with_info=True,as_supervised=True) 
train_data,test_data = imdb['train'],imdb['test']
# you may download tokenized ones (even ones with subword tokenization): "imdb_reviews/subwords8k" 
# then to initialize tokenizer object do: tokenizer = info.features['text'].encoder
# finally you can do: tokenizer.encode(some_string) or tokenizer.decode(some_tokenized_string)
# and skip the following steps until Training

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
# you can add maxlen=int to define or limit the length of sequences. default is the longest sentence. 
# you can add padding='post', default is 'pre'. you can truncate='post' to remove from the end, default is 'pre' again
padded_seq = pad_sequences(sequences,maxlen=max_length) 

sequences_test = tokenizer.texts_to_sequences(testing_sentences)
padded_seq_test = pad_sequences(sequences_test,maxlen=max_length) 

# =============================================================================
# Train 
# 
# for subword learning, sequence is very important. So, make sure to learn the sequence and order.
# (see https://keras.io/api/layers/core_layers/embedding/)
# https://www.coursera.org/learn/natural-language-processing-tensorflow/supplement/oHNdd/try-it-yourself
# 
# To see good examples, visit: https://www.coursera.org/learn/natural-language-processing-tensorflow/supplement/TAAsf/exploring-different-sequence-models
# 
# =============================================================================
adam = tf.keras.optimizers.Adam(lr=0.01)

    # =============================================================================
    # Dense
    # =============================================================================
model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
        tf.keras.layers.Flatten(), #or tf.keras.layers.GlobalAveragePooling1D()
        tf.keras.layers.Dense(6,activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid'),
     ])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


del model
inputs = tf.keras.Input(shape=(max_length,), dtype="int32")
x = tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length)(inputs)
# use tf.keras.layers.GlobalAveragePooling1D() specifically if using weird shapes as a result of subword tokens etc.
x = tf.keras.layers.Flatten()(x) 
x = tf.keras.layers.Dense(6,activation='relu')(x)
outputs = tf.keras.layers.Dense(1,activation='sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

    # =============================================================================
    # LSTM
    # =============================================================================
    # https://keras.io/examples/nlp/bidirectional_lstm_imdb/
del model
inputs = tf.keras.Input(shape=(None,), dtype="int32")
x = tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length)(inputs)
# using bidirectional LSTM instead of flatten or averaging can improve the results
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
x = tf.keras.layers.Dense(6,activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

    # =============================================================================
    # GRU
    # =============================================================================
    # https://keras.io/examples/nlp/bidirectional_lstm_imdb/
del model
inputs = tf.keras.Input(shape=(None,), dtype="int32")
x = tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length)(inputs)
# using bidirectional LSTM instead of flatten or averaging can improve the results
x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64))(x) # Gate recurrent units
x = tf.keras.layers.Dense(6,activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

    # =============================================================================
    # Conv
    # =============================================================================
    # https://www.coursera.org/learn/natural-language-processing-tensorflow/lecture/fSE8o/using-a-convolutional-network
del model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


del model
inputs = tf.keras.Input(shape=(None,), dtype="int32")
x = tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length)(inputs)
# using bidirectional LSTM instead of flatten or averaging can improve the results
x = tf.keras.layers.Conv1D(128,5,activation='relu')(x) # 128 neurons with kernel size of 5 words
x = tf.keras.layers.GlobalMaxPooling1D()(x)
x = tf.keras.layers.Dense(24,activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

    # =============================================================================
    # Multi embedding
    # =============================================================================
col1, col2 = tf.keras.Input(shape=(10,)), tf.keras.Input(shape=(10,))
col1_embeded = tf.keras.layers.Embedding(500, 64)(col1)
col2_embedded = tf.keras.layers.Embedding(100, 64)(col2)
#...

# =============================================================================input_sequences
# Fit model
# =============================================================================
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=35)
checkpoint = tf.keras.callbacks.ModelCheckpoint('models/pretrain_best_model-cocite_.h5', monitor='val_accuracy', mode='min', save_best_only=True)
tensorboard = tf.keras.callbacks.TensorBoard(
                          log_dir='.\logs',
                          histogram_freq=1,
                          write_images=True
                        )

history = model.fit(padded_seq, 
                      training_labels_final, 
                      epochs=num_epochs, 
                      validation_data=(padded_seq_test, testing_labels_final),
                      callbacks=[callback, checkpoint,tensorboard])

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

# =============================================================================
# Get word embeddings
# =============================================================================
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

# =============================================================================
# Prepare vector format for Tensorflow projector
# =============================================================================
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
print(decode_review(padded_seq_test[3]))
print(training_sentences[3])

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
