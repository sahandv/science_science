#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:46:31 2021

@author: github.com/sahandv
"""
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras 
# from tensorflow.keras.preprocessing.text import Tokenizer,WordpieceTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tokenizers import Tokenizer,BertWordPieceTokenizer, models, pre_tokenizers, decoders, trainers, processors
import matplotlib.pyplot as plt
from tqdm import tqdm
from sciosci.assets import text_assets as ta
from sciosci.assets import ann_assets as anna
from sklearn.model_selection import train_test_split
# !wget 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt'

tokenizer = 'word'

# =============================================================================
## Next Word Prediction task
# https://www.coursera.org/learn/natural-language-processing-tensorflow/lecture/B80b0/notebook-for-lesson-1
# 
# For character prediction see
# https://www.tensorflow.org/tutorials/text/text_generation
# =============================================================================
dir_root = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/cora-classify/cora/' # C1314
# dir_root = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/Corpus/cora-classify/cora/' # Ryzen

    # =============================================================================
    # Load and prepare features
    # =============================================================================
corpus_idx = pd.read_csv(dir_root+'clean/single_component_small_18k/corpus_idx_original')['id'].values.tolist()
net_vecs = pd.read_csv(dir_root+'embeddings/single_component_small_18k/n2v 300-70-20 p1q05').drop('Unnamed: 0',axis=1)
net_vecs.columns = ['net_cid_'+str(x) for x in range(len(net_vecs.columns))]

data_path_rel = dir_root+'extractions_with_unique_id_labeled_single_component.csv'
data = pd.read_csv(data_path_rel)
data = data[data['id'].isin(corpus_idx)]
    # =============================================================================
    # Load and tokenize text
    # =============================================================================
corpus = pd.read_csv(dir_root+'clean/single_component_small_18k/abstract_title super duper pure',names=['abstract'])
corpus['abstract'] = "[documentembeddingtoken] "+corpus['abstract'] 
# corpus.to_csv(dir_root+'clean/single_component_small_18k/abstract_title super duper pure with [DOC]',header=False,index=False)
corpus = corpus['abstract'].values.tolist()

text_lens = np.array([len(p.split()) for p in corpus])
max_paragraph_len = int(np.percentile(text_lens, 95)) # take Nth percentile as the sentence length threshold

# corpus = [
#     '1 red brown fox',
#     'red brown cat!',
#     'red cat loves chicken wings.',
#     'black cat loves chicken wings as well',
#     'brown fox hates cats'
#     ]

embedding_dim = 128
num_epochs = 500
vocab_limit = 30000
n_classes = vocab_limit
n_docs = len(corpus_idx)+1

if tokenizer=='word':
##################
# TF word tokenize
##################
    # oov is out of vocabulary token replacement. you can num_words=int
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='[OOV_TKN]',num_words=vocab_limit)#(oov_token='<oov_tkn>') 
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index)+1
    word_index = tokenizer.word_index
    print(total_words)
    # print(tokenizer.word_index)
    print(tokenizer.word_index['documentembeddingtoken'])
    
    n = 6
    ##################
    # unlimited length
    ##################
    # extract n-gram sequences from n=2 to n=number_of_grams_in_sentences 
    input_sequences = []
    input_sequences_doc_id = []
    for cid,sent in tqdm(enumerate(corpus),total=len(corpus)):
        token_list = tokenizer.texts_to_sequences([sent])[0]
        for i in range(n,len(token_list)):
            n_gram_sentence = token_list[:i+1]
            input_sequences.append(n_gram_sentence)   
            input_sequences_doc_id.append(cid)
    ##################
    # limited length
    ##################
    # OR extract n-gram sequences from n=2 to n=min(number_of_grams_in_sentences,max_paragraph_len )
    input_sequences = []
    input_sequences_doc_id = []
    for cid,sent in tqdm(enumerate(corpus),total=len(corpus)):
        token_list = tokenizer.texts_to_sequences([sent])[0]
        for i in range(n,min(len(token_list),max_paragraph_len)):
            n_gram_sentence = token_list[:i+1]
            input_sequences.append(n_gram_sentence)
            input_sequences_doc_id.append(cid)


if tokenizer=='bpe':
##################
# BPE tokenize
##################
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    trainer = trainers.BpeTrainer(vocab_size=20000, min_frequency=2,special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[documentembeddingtoken]"])
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

if tokenizer=='bert':
##################
# BERT WP tokenize
##################
    tokenizer = BertWordPieceTokenizer("data/vocabs/bert-base-uncased-vocab.txt", lowercase=True)



    # =============================================================================
    # Prepare sequences
    # =============================================================================
print("Preparing sequences")
revers_word_index = ta.reverse_word_index(word_index)

# you can add maxlen=int. you can add padding='post', default is 'pre'. you can truncate='post' to remove from the end, default is 'pre' again
max_seq_len = max([len(x) for x in input_sequences])
# max_seq_len = int(max_paragraph_len)
input_sequences = pad_sequences(input_sequences,maxlen=max_seq_len,padding='pre')
input_sequences = np.array(input_sequences)
input_sequences_tmp = input_sequences[:3000,:]

    # =============================================================================
    # Prepare model inputs and outputs
    # =============================================================================
# create X and Y
X,labels = input_sequences[:,:-1],input_sequences[:,-1]
input_df = pd.DataFrame(X)
input_df['Y'] = labels
input_df['corpus_index'] = input_sequences_doc_id
input_df_sample = input_df.sample(10)
x2 = net_vecs.values

# shuffle dataset
input_df = input_df.sample(frac=1)

# split train test
msk = np.random.rand(len(input_df)) < 0.8
train = input_df[msk]
test = input_df[~msk]


train_corpus_idx = train['corpus_index'].values
train_y = train['Y'].values
# train_y_cat = tf.keras.utils.to_categorical(train_y, num_classes=n_classes)
train_x1 = train[list(range(max_seq_len-1))].values
train_x2 = pd.DataFrame(train['corpus_index']).reset_index().merge(net_vecs.reset_index(),on='index',how='left').drop('index',axis=1).drop('corpus_index',axis=1).values

test_corpus_idx = test['corpus_index'].values
test_y = test['Y'].values
# test_y_cat = tf.keras.utils.to_categorical(test_y, num_classes=n_classes)
test_x1 = test[list(range(max_seq_len-1))].values
test_x2 = pd.DataFrame(test['corpus_index']).reset_index().merge(net_vecs.reset_index(),on='index',how='left').drop('index',axis=1).drop('corpus_index',axis=1).values

# corpus_idx = train_corpus_idx
# y = train_y

# Xtrain, Xtest, Labeltrain, Labeltest = train_test_split(X, labels, test_size=0.2, random_state=100,shuffle=True)
# ys = tf.keras.utils.to_categorical(labels,num_classes=total_words) # Not suitable for large data

# del train, test, msk, input_df, X, x2
gc.collect()

##################
# Generator functional
##################

def _input_fn(x1,x2,y=None,n_classes:int=None,corpus_idx=None,batch_size=2):
    """
    Generator function for tensorflow.    

    Parameters
    ----------
    x1 : np.array
    x2 : np.array
    y : np.array, optional
    corpus_idx : np.array, optional
        Will use this id to map x2 to x1 dimension. This is intended to conserve space.
    n_classes: int
        Number of classes. This is usually the vocabulary size
    Returns
    -------
    Generator function.

    """
    # if corpus_idx!=None:
        # Map x2 to x1 dims
        # x2 = corpus_idx.merge(x2,on='index',how='left').drop('index',axis=1)
    
    def generator():
        for i in range(x1.shape[0]):
            yield {"input_1":x1[i],"input_2":x2[corpus_idx[i]]}, np.eye(n_classes)[y[i]]
    
    dataset = tf.data.Dataset.from_generator(generator, output_types=({"input_1": tf.int64, "input_2": tf.float64}, tf.int64))
    # dataset = tf.data.Dataset.from_generator(generator,output_signature=(
    #     {"input_1":tf.TensorSpec(shape=(x1.shape[1],), dtype=tf.float32),
    #      "input_2":tf.TensorSpec(shape=(x2.shape[1],), dtype=tf.float32)},
    #     tf.TensorSpec(shape=(n_classes,), dtype=tf.float32),
    #     ))
    dataset = dataset.batch(batch_size)
    return dataset

##################
# Generator object - updated - reference https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
##################
class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df, X_col, y_col,
                 batch_size,
                 input_size=(224, 224, 3),
                 shuffle=True):
        
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
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


##################
# Generator object
##################
class DataGenerator(tf.keras.utils.Sequence):
    """
    Generates data for Keras
    Sequence based data generator.
    """
    def __init__(self,x1,x2,y=None,n_classes:int=None,corpus_idx=None,batch_size=2,to_fit=True):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.n_classes = n_classes
        self.corpus_idx = corpus_idx
        self.batch_size = batch_size
        self.to_fit = to_fit
        self.ids = list(range(len(x1)))
        
    def __len__(self):
        return int(np.floor(self.x1.shape[0] / self.batch_size))
    
    def __getitem__(self, index):
        """
        Generate one batch of data
        
        Parameters
        -------
        index: index of the batch
        
        Returns
        -------
        X and y when fitting. X only when predicting
        """
        ID_list = self.ids[index * self.batch_size:(index + 1) * self.batch_size]
        
        X1 = self._generate_X1(ID_list)
        X2 = self._generate_X2(ID_list)
        
        if self.to_fit:
            y = self._generate_y(ID_list)
            return [X1,X2], y
        else:
            return [X1,X2]
        
    def _generate_y(self, ID_list):
        y_batch = np.empty((self.batch_size), dtype=int)
        for i in ID_list:
            y_batch[i] = self.y[i]
        return tf.keras.utils.to_categorical(y_batch, num_classes=self.n_classes) 
        
    def _generate_X1(self, ID_list):
        x_batch = np.empty((self.batch_size,self.x1.shape[1]), dtype=int)
        for i in ID_list:
            x_batch[i,] = self.x1[i]
        return(x_batch) 
       
    def _generate_X2(self, ID_list):
        x_batch = np.empty((self.batch_size,self.x2.shape[1]), dtype=int)
        for i in ID_list:
            x_batch[i,] = self.x2[self.corpus_idx[i]]
        return(x_batch)

train_dataset = DataGenerator(x1=train_x1, x2=x2,y=train_y,n_classes=n_classes,corpus_idx=train_corpus_idx,batch_size=1)
valid_dataset = DataGenerator(x1=test_x1, x2=x2,y=test_y,n_classes=n_classes,corpus_idx=test_corpus_idx,batch_size=1)

##################
# Generator tf.data
##################
def y_generator(y:np.array,n_classes:int):
    for label in y:
        yield tf.keras.utils.to_categorical(y,n_classes)
        
def x_generator(x,x2,corpus_idx):
    for i,idx in enumerate(corpus_idx):
        yield [x[i],x2[idx]]

train_x_dataset = tf.data.Dataset.from_generator(x_generator,output_types=(tf.float32,tf.float32),output_shapes=((None,train_x1.shape[1]),(None,x2.shape[1])),args=[train_x1,x2,train_corpus_idx])
train_y_dataset = tf.data.Dataset.from_generator(y_generator,output_types=tf.int32,output_shapes=((None,n_classes)),args=[train_y,n_classes])
train_dataset = tf.data.Dataset.zip((train_x_dataset,train_y_dataset))


valid_x_dataset = tf.data.Dataset.from_generator(x_generator,output_types=(tf.float32,tf.float32),output_shapes=((None,train_x1.shape[1]),(None,x2.shape[1])),args=[test_x1,x2,test_corpus_idx])
valid_y_dataset = tf.data.Dataset.from_generator(y_generator,output_types=tf.int32,output_shapes=((None,n_classes)),args=[test_y,n_classes])
valid_dataset = tf.data.Dataset.zip((valid_x_dataset,valid_y_dataset))

del msk, input_df, train, test
gc.collect()

train_dataset = train_dataset.shuffle(5000).batch(32)
valid_dataset = valid_dataset.shuffle(5000).batch(32)

##################
# Generator tf.data
##################
def generator(x1,x2,corpus_idx,y,n_classes):
    for i in range(x1.shape[0]):
        yield {"input_1":x1[i],"input_2":x2[corpus_idx[i]]}, np.eye(n_classes)[y[i]]

train_dataset = tf.data.Dataset.from_generator(generator,args=[train_x1,x2,train_corpus_idx,train_y,n_classes],output_types=({"input_1": tf.float32, "input_2": tf.float32}, tf.int8))
train_dataset = train_dataset.batch(32)

test_dataset = tf.data.Dataset.from_generator(generator,args=[test_x1,x2,test_corpus_idx,test_y,n_classes],output_types=({"input_1": tf.float32, "input_2": tf.float32}, tf.int8))
test_dataset = test_dataset.batch(32)
    # =============================================================================
    # Train
    # =============================================================================
inputs_seq = tf.keras.Input(shape=(None,), dtype="int32",name='input_1')
inputs_doc = tf.keras.Input(shape=(1,), dtype="int32",name='input_2')
inputs_netvec = tf.keras.Input(shape=(300,), dtype="int32",name='input_3')
# inputs_aux= tf.keras.Input(shape=(3,), dtype="int32",name='features input')

x_11 = tf.keras.layers.Embedding(n_classes,embedding_dim,input_length=max_seq_len-1)(inputs_seq)
x_11 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150,return_sequences=True))(x_11)
x_11 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150))(x_11)
x_11 = tf.keras.Model(inputs=inputs_seq, outputs=x_11)

x_12 = tf.keras.layers.Embedding(n_docs,embedding_dim,input_length=1)(inputs_doc)
x_12 = tf.keras.layers.Flatten()(x_12) 
x_12 = tf.keras.layers.Dense(300,activation='relu')(x_12)
x_12 = tf.keras.Model(inputs=inputs_doc, outputs=x_12)

x_2 = tf.keras.layers.Dense(100,activation='relu')(inputs_netvec)
x_2 = tf.keras.Model(inputs=inputs_netvec, outputs=x_2)

# x_3 = tf.keras.layers.Dense(100,activation='relu')(inputs_aux)
# x_3 = tf.keras.Model(inputs=inputs_aux, outputs=x_3)

x = tf.keras.layers.concatenate([x_11.output, x_12.output, x_2.output], name='combination')
x = tf.keras.layers.Dense(400,activation='relu')(x)
x = tf.keras.layers.Dense(100,activation='relu')(x)
outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
model = keras.Model(inputs=[x_11.input,x_12.input, x_2.input], outputs=outputs)

adam = tf.keras.optimizers.Adam(lr=0.01)

model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
model.summary()
tf.keras.utils.plot_model(model, to_file='combined_embedding.png', show_shapes=True, show_layer_names=True)

callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy',patience=35)
checkpoint = tf.keras.callbacks.ModelCheckpoint('./models/pretrain_next_word_pred.h5', monitor='accuracy', mode='min', save_best_only=True)
tensorboard = tf.keras.callbacks.TensorBoard(
                          log_dir='.\logs',
                          histogram_freq=1,
                          write_images=True
                        )

history = model.fit(_input_fn(train_x1, x2,train_y,n_classes=n_classes,corpus_idx=train_corpus_idx,batch_size=2),
                    epochs=num_epochs, 
                    validation_data=_input_fn(test_x1, x2,test_y,n_classes=n_classes,corpus_idx=test_corpus_idx,batch_size=2),
                    verbose=1,
                    callbacks=[callback, checkpoint,tensorboard])


history = model.fit(train_dataset,
                    epochs=num_epochs, 
                    validation_data=test_dataset,
                    verbose=1,
                    callbacks=[callback, checkpoint,tensorboard])

history = model.fit([train_x1,train_corpus_idx,train_x2],train_y_cat,
                    epochs=num_epochs, 
                    validation_data=([test_x1,test_corpus_idx,test_x2],test_y_cat),
                    verbose=1,
                    callbacks=[callback, checkpoint,tensorboard])


    # =============================================================================
    # Result vis
    # =============================================================================
anna.plot_graphs(history,'accuracy')

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