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


# =============================================================================
# Train Model
# =============================================================================
texts = pd.read_csv('/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/Corpus/Unsupervised Training/Docs/patent_wos',names=['abstract'])['abstract'].values.tolist()
documents = [TaggedDocument(doc.lower().split(), [i]) for i, doc in enumerate(texts)]
model = Doc2Vec(documents, size=100, window=5, min_count=1, max_count=12, workers=15,epochs=10)
fname = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/Doc2Vec Models/patent_wos corpus'
model.save(fname)


# =============================================================================
# Test Model
# =============================================================================
fname = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Doc2Vec Models/deflemm/patent_wos corpus'
model = Doc2Vec.load(fname)

fname = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/KPRIS/clean/abstract_title deflemm'
texts = pd.read_csv(fname,names=['abstract'])['abstract'].values.tolist()
test_docs = [doc.lower().split() for doc in texts]

start_alpha=0.01
infer_epoch=1000
X=[]
for d in tqdm(test_docs):
    X.append( model.infer_vector(d, alpha=start_alpha, steps=infer_epoch))
X_df = pd.DataFrame(X)
X_df.to_csv('/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/KPRIS/embeddings/deflemm/Doc2Vec patent_wos corpus',index=False)