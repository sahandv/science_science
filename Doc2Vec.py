#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:13:50 2020

@author: sahand
"""

import pandas as pd
# from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm


# dir_root = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/'
dir_root = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/'

# =============================================================================
# Train Model
# =============================================================================
texts = pd.read_csv(dir_root+'Corpus/Dimensions/abstract_title deflemm',names=['abstract'])['abstract'].values.tolist()
documents = [TaggedDocument(doc.lower().split(), [i]) for i, doc in enumerate(texts)]
model = Doc2Vec(documents, size=128, window=10, min_count=1, dm=1, workers=16, epochs=40)
fname = dir_root+'Corpus/Dimensions/models/doc2vec 128D dm=1 window=10'
model.save(fname)


# =============================================================================
# Test Model
# =============================================================================
fname = dir_root+'Corpus/Dimensions/models/doc2vec 128D dm=1 window=10'
model = Doc2Vec.load(fname)

fname = dir_root+'Corpus/Dimensions/abstract_title deflemm'
texts = pd.read_csv(fname,names=['abstract'])['abstract'].values.tolist()
test_docs = [doc.lower().split() for doc in texts]
test_docs = test_docs[480000:]


start_alpha=0.01
infer_epoch=1000
X=[]
for d in tqdm(test_docs):
    X.append( model.infer_vector(d, alpha=start_alpha, steps=infer_epoch))
X_df = pd.DataFrame(X)
X_df.to_csv(dir_root+'Corpus/Dimensions/embeddings/doc2vec 128D dm=1 window=10 480000:',index=False)