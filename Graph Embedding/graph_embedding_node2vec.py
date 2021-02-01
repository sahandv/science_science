#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:27:46 2020

@author: sahand
"""
import sys
import gc
import warnings
from text_unidecode import unidecode
from collections import deque
warnings.filterwarnings('ignore')


import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm
# conda install -n base -c conda-forge widgetsnbextension
# conda install -c conda-forge ipywidgets
from node2vec import Node2Vec
from gensim.models import Word2Vec

sns.set_style('whitegrid')
seed=1
np.random.seed(seed)
# =============================================================================
# Read data
# =============================================================================
# dir_path = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/Corpus/cora-classify/cora/'
dir_path = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/Corpus/Dimensions/'
data = pd.read_csv(dir_path+'citations pairs')#, names=['referring_id','cited_id'],sep='\t')
data.columns = ['referring_id','cited_id']
gc.collect()
sample = data.sample()
data.info(memory_usage='deep')
idx = pd.read_csv(dir_path+'publication idx',names=['id'])#(dir_path+'corpus idx',index_col=0)
idx.columns = ['id']
idx['id'] = idx['id'].str.replace('pub.','').astype(str).astype(int)
idx = idx['id'].values.tolist()

# =============================================================================
# Prepare graph
# =============================================================================
graph = nx.Graph()
for i,row in tqdm(data.iterrows(),total=data.shape[0]):
    graph.add_edge(row['referring_id'],row['cited_id'])

print('Graph fully connected:',nx.is_connected(graph))
print('Connected components:',nx.number_connected_components(graph))

# connected_components = list(nx.connected_components(graph))

del data
gc.collect()
# =============================================================================
# Train
# =============================================================================
node2vec = Node2Vec(graph, dimensions=128, walk_length=80, num_walks=10, workers=1, p=1, q=1,seed=seed)
model = node2vec.fit(window=10, min_count=1)
model.save(dir_path+'models/node2vec deepwalk-128-80-10 p1q1')

# =============================================================================
# Get embeddings
# =============================================================================
model_name = 'node2vec deepwalk-128-80-10 p1q1'
model = Word2Vec.load(dir_path+'models/'+model_name)
embeddings = []
idx_true = []
miss_count = 0
for i in tqdm(idx):
    # embeddings.append(model.wv[str(i)])
    try:
        embeddings.append(model.wv[str(i)])
        idx_true.append(i)
    except:
        miss_count+=1
        print('Error while getting the embedding',i,':',sys.exc_info()[0])
print('total misses:',miss_count)

embeddings = pd.DataFrame(embeddings)
embeddings.index = idx_true
embeddings.to_csv(dir_path+'embeddings/'+model_name,index=True)


