#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:27:46 2020

@author: sahand
"""

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

sns.set_style('whitegrid')

# =============================================================================
# Read data
# =============================================================================
dir_path = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/Corpus/cora-classify/cora/'
data = pd.read_csv(dir_path+'citations_filtered.csv')#, names=['referring_id','cited_id'],sep='\t')

# =============================================================================
# Prepare graph
# =============================================================================
graph = nx.Graph()
for i,row in tqdm(data.iterrows(),total=data.shape[0]):
    graph.add_edge(row['referring_id'],row['cited_id'])

# =============================================================================
# Train
# =============================================================================
node2vec = Node2Vec(graph, dimensions=100, walk_length=100, num_walks=18, workers=2)
model = node2vec.fit(window=10, min_count=1)
model.save(dir_path+'models/node2vec-100-18-100')

# =============================================================================
# Get embeddings
# =============================================================================
