#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:46:31 2021

@author: github.com/sahandv
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras 
# from tensorflow.keras.preprocessing.text import Tokenizer,WordpieceTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tokenizers import Tokenizer,BertWordPieceTokenizer, models, pre_tokenizers, decoders, trainers, processors
import matplotlib.pyplot as plt
from tqdm import tqdm


# !wget 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt'

# =============================================================================
## Next Word Prediction task
# https://www.coursera.org/learn/natural-language-processing-tensorflow/lecture/B80b0/notebook-for-lesson-1
# 
# For character prediction see
# https://www.tensorflow.org/tutorials/text/text_generation
# =============================================================================

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
    # =============================================================================
    # prepare 
    # =============================================================================
# dir_root = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/cora-classify/cora/' # C1314
dir_root = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/Corpus/cora-classify/cora/' # Ryzen

corpus = pd.read_csv(dir_root+'clean/single_component_small_18k/abstract_title super duper pure',names=['abstract'])
corpus['abstract'] = "[DOC] "+corpus['abstract'] 
corpus = corpus['abstract'].values.tolist()
data_path_rel = dir_root+'extractions_with_unique_id_labeled_single_component.csv'
data = pd.read_csv(data_path_rel)

text_lens = np.array([len(p.split()) for p in corpus])
max_paragraph_len = int(np.percentile(text_lens, 95))

# corpus = [
#     '1 red brown fox',
#     'red brown cat!',
#     'red cat loves chicken wings.',
#     'black cat loves chicken wings as well',
#     'brown fox hates cats'
#     ]
embedding_dim = 128
num_epochs = 500

##################
# word tokenize
##################
# oov is out of vocabulary token replacement. you can num_words=int
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='[OOV_TKN]')#(oov_token='<oov_tkn>') 
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index)+1

index = tokenizer.word_index
print(total_words)
# print(tokenizer.word_index)
print(tokenizer.word_index['doc_embedding'])

# extract n-gram sequences from n=2 to n=number_of_grams_in_sentences 
input_sequences = []

for sent in tqdm(corpus):
    token_list = tokenizer.texts_to_sequences([sent])[0]
    for i in range(1,len(token_list)):
        n_gram_sentence = token_list[:i+1]
        input_sequences.append(n_gram_sentence)


# OR extract n-gram sequences from n=2 to n=min(number_of_grams_in_sentences,max_paragraph_len )
input_sequences = []
for sent in tqdm(corpus):
    token_list = tokenizer.texts_to_sequences([sent])[0]
    for i in range(1,min(len(token_list),max_paragraph_len)):
        n_gram_sentence = token_list[:i+1]
        input_sequences.append(n_gram_sentence)



##################
# BPE tokenize
##################
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
trainer = trainers.BpeTrainer(vocab_size=20000, min_frequency=2,special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[DOC]"])
tokenizer.train([
# 	"./path/to/dataset/1.txt",
# 	"./path/to/dataset/2.txt",
	dir_root+'clean/single_component_small_18k/abstract_title super duper pure'
], trainer=trainer)
tokenizer.save(dir_root+"clean/single_component_small_18k/byte-level-bpe.tokenizer.json", pretty=True)

tokenizer = Tokenizer.from_file(dir_root+"clean/single_component_small_18k/byte-level-bpe.tokenizer.json")
output = tokenizer.encode_batch(["I can feel the magic, can you?"])
print(output[0].tokens)
print(output[0].ids)

##################
# BERT WP tokenize
##################
tokenizer = BertWordPieceTokenizer("data/vocabs/bert-base-uncased-vocab.txt", lowercase=True)




# you can add maxlen=int. you can add padding='post', default is 'pre'. you can truncate='post' to remove from the end, default is 'pre' again
max_seq_len = max([len(x) for x in input_sequences])
# max_seq_len = int(max_paragraph_len)
input_sequences = pad_sequences(input_sequences,maxlen=max_seq_len,padding='pre') 
input_sequences = np.array(input_sequences)
input_sequences_tmp = input_sequences[:30,:]

# create X and Y
xs,labels = input_sequences[:,:-1],input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels,num_classes=total_words)

    # =============================================================================
    # Train
    # =============================================================================
inputs = tf.keras.Input(shape=(None,), dtype="int32")
inputs = tf.keras.Input(shape=(None,), dtype="int32")
x = tf.keras.layers.Embedding(total_words,embedding_dim,input_length=max_seq_len-1)(inputs)
# x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150,return_sequences=True))(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150))(x)
# x = tf.keras.layers.Dense(6,activation='relu')(x)
outputs = tf.keras.layers.Dense(total_words, activation="softmax")(x)
model = keras.Model(inputs, outputs)

adam = tf.keras.optimizers.Adam(lr=0.01)

model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy',patience=35)
checkpoint = tf.keras.callbacks.ModelCheckpoint('./models/pretrain_next_word_pred.h5', monitor='accuracy', mode='min', save_best_only=True)
tensorboard = tf.keras.callbacks.TensorBoard(
                          log_dir='.\logs',
                          histogram_freq=1,
                          write_images=True
                        )

history = model.fit(xs, ys,
                    epochs=num_epochs, 
                    # validation_data=(padded_seq_test, testing_labels_final),
                    verbose=1,
                    callbacks=[callback, checkpoint,tensorboard])

    # =============================================================================
    # Result vis
    # =============================================================================
plot_graphs(history,'accuracy')

    # =============================================================================
    # Test and predict 
    # =============================================================================
seed_text = 'black cat hates'
next_words = 10


for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list],maxlen=max_seq_len-1,padding='pre')
    # predicted = model.predict_classes(token_list,verbose=0) # only for sequential model
    predicted = np.argmax(model.predict(token_list,verbose=0),axis=1)
    output_word = ""
    for word,index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text+= " " + output_word
    
    print(seed_text)