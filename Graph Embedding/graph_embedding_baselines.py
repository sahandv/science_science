#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:22:36 2021

@author: github.com/sahandv
"""
import sys
import gc
import pandas as pd
import numpy as np
import networkx as nx
import karateclub as kc
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

# =============================================================================
# Init Cora
# =============================================================================
dir_root = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/Corpus/cora-classify/cora/'
texts = pd.read_csv(dir_root+'clean/with citations new/abstract_title all-lem',names=['abstract'])['abstract'].values.tolist()
networks = pd.read_csv(dir_root+'citations_filtered.csv')# with_d2v300D_supernodes.csv')#, names=['referring_id','cited_id'],sep='\t')
networks.columns = ['referring_id','cited_id']

gc.collect()
sample = networks.sample()
networks.info(memory_usage='deep')
idx = pd.read_csv(dir_root+'clean/with citations new/corpus idx')#,names=['id'])#(dir_path+'corpus idx',index_col=0)
idx.columns = ['id']
# idx['id'] = idx['id'].str.replace('pub.','').astype(str).astype(int)
idx = idx['id'].values.tolist()

# =============================================================================
# Prepare graph
# =============================================================================
networks = networks[(networks['referring_id'].isin(idx)) | (networks['cited_id'].isin(idx))] # mask

graph = nx.Graph()
for i,row in tqdm(networks.iterrows(),total=networks.shape[0]):
    graph.add_edge(row['referring_id'],row['cited_id'])

print('Graph fully connected:',nx.is_connected(graph))
print('Connected components:',nx.number_connected_components(graph))

# connected_components = list(nx.connected_components(graph))

del networks
gc.collect()
# =============================================================================
# Text embedding (TFIDF)
# =============================================================================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
X.shape
# =============================================================================
# TADW
# =============================================================================
TADW = kc.node_embedding.TADW(dimensions=120, reduction_dimensions=240,lambd=1)
model_TADW = TADW.fit(graph,X)
TADW_vectors = model_TADW.get_embedding() 
# =============================================================================
# TENE
# =============================================================================
TENE = kc.node_embedding.TENE()
model_TENE = TENE.fit(graph,X)
TENE_vectors = model_TENE.get_embedding() 

