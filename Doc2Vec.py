#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:13:50 2020

@author: github.com/sahandv
"""
import gc
import pandas as pd
import numpy as np
# from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from itertools import chain
from scipy import spatial,sparse,sign

from_data = 450000
to_data = 500000
dir_root = '/home/sahand/GoogleDrive/Data/'
# dir_root = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/'
texts = pd.read_csv(dir_root+'Corpus/Dimensions All/clean/abstract_title method_b_3')['abstract'].values.tolist()[from_data:to_data]

# shared_tags = pd.read_csv(dir_root+'Corpus/cora-classify/cora/embeddings/single_component_small_18k/n2v 300-70-20 p1q05 DEC 500, 1000, 1000, 500, 10 k10 labels - 0')['label'].values.tolist()
# =============================================================================
# Train Model
# =============================================================================
# documents = [TaggedDocument(word_tokenize(doc.lower()), [i]) for i, doc in enumerate(texts)]
# model = Doc2Vec(documents, vector_size=300, window=10, min_count=1, dm=1, workers=15, epochs=40)
# fname = dir_root+'Corpus/Dimensions All/models/doc2vec 300D dm=1 window=10 b3'
# model.save(fname)

# =============================================================================
# Train Model with Tags
# =============================================================================
# tagged_documents = [TaggedDocument(words=word_tokenize(_d.lower()), tags=['cluster_'+str(shared_tags[i]),str(i)]) for i, _d in enumerate(texts)]
# model = Doc2Vec(tagged_documents, size=300, window=10, min_count=1, dm=1, workers=16, epochs=40)
# fname = dir_root+'Corpus/Dimensions All/models/doc2vec 300D dm=1 window=10 tagged'
# model.save(fname)


# =============================================================================
# Test Model
# =============================================================================
fname = dir_root+'Corpus/Dimensions All/models/doc2vec 300D dm=1 window=10 b3'
model = Doc2Vec.load(fname)
documents = [word_tokenize(doc.lower()) for doc in tqdm(texts)]
# test_docs2 = [doc.lower().split() for doc in texts] # This is way faster than word_tokenize
# test_docs = test_docs[480000:]


start_alpha=0.01
infer_epoch=1000
X=[]
for d in tqdm(documents):
    X.append( model.infer_vector(d, alpha=start_alpha, epochs=infer_epoch))
X_df = pd.DataFrame(X)
X_df.to_csv(dir_root+'Corpus/Dimensions All/embeddings/doc2vec 300D dm=1 window=10 b3 '+str(from_data),index=False)

#%%### =============================================================================
# concat vecs
# =============================================================================

sections = ['200000','250000','300000','350000','400000','450000','500000']

data = pd.read_csv(dir_root+'Corpus/Dimensions All/embeddings/doc2vec 300D dm=1 window=10 b3 0')
for section in tqdm(sections):
    data = data.append(pd.read_csv(dir_root+'Corpus/Dimensions All/embeddings/doc2vec 300D dm=1 window=10 b3 '+section),ignore_index=True)

data = data.reset_index(drop=True)
data.to_csv(dir_root+'Corpus/Dimensions All/embeddings/doc2vec 300D dm=1 window=10 b3',index=False)

# =============================================================================
# Get keyword embedding
# =============================================================================
directory = dir_root+'Corpus/Taxonomy/'
file_name = 'CSO.3.3-with-labels-US-lem.csv'
corpus = pd.read_csv(directory+file_name)
corpus = corpus.a.values.tolist()+corpus.b.values.tolist()
keywords = list(set(corpus))

fname = dir_root+'Corpus/Dimensions All/models/doc2vec 300D dm=1 window=10 b3'
model = Doc2Vec.load(fname)

vectors = []
for keyword in tqdm(keywords,total=len(keywords)):
    tokens = keyword.split(' ')
    keyword_vec = []
    if len(tokens)>1:
        for token in tokens:
            if len(token)>1:
                keyword_vec.append(model.wv['neural'])
        if len(keyword_vec)>0:
            keyword_vec = np.array(keyword_vec).mean(axis=0)
        else:
            keyword_vec = np.zeros(100)
            keyword_vec[:] = np.nan
    else:
        keyword_vec = model.wv['neural']
    vectors.append(keyword_vec)
vectors = pd.DataFrame(vectors)
vectors

# vec_a = model.infer_vector(['machine','learning'])
vec_a = np.array([model['reinforcement'],model['learning']]).mean(axis=0)
vec_b = np.array([model['s'],model['learning']]).mean(axis=0)

distance_tmp = spatial.distance.cosine(vec_a, vec_b)
similarity_tmp = 1 - distance_tmp
similarity_tmp
