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
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer,CountVectorizer

# =============================================================================
# Init Cora
# =============================================================================
dir_root = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/Corpus/cora-classify/cora/'
texts = pd.read_csv(dir_root+'clean/single_component/abstract_title all-lem',names=['abstract'])['abstract'].values.tolist()
labels = pd.read_csv(dir_root+'clean/single_component/corpus classes1')['class1'].values.tolist()
networks = pd.read_csv(dir_root+'citations_filtered_single_component.csv')# with_d2v300D_supernodes.csv')#, names=['referring_id','cited_id'],sep='\t')
networks.columns = ['referring_id','cited_id']

gc.collect()
sample = networks.sample()
networks.info(memory_usage='deep')
idx = pd.read_csv(dir_root+'clean/single_component/corpus idx')#,names=['id'])#(dir_path+'corpus idx',index_col=0)
idx.columns = ['id']
# idx['id'] = idx['id'].str.replace('pub.','').astype(str).astype(int)
idx = idx['id'].values.tolist()
networks = networks[(networks['referring_id'].isin(idx)) & (networks['cited_id'].isin(idx))] # mask
# =============================================================================
# component selector
# =============================================================================
graph = nx.Graph()
for i,row in tqdm(networks.iterrows(),total=networks.shape[0]):
    graph.add_edge(row['referring_id'],(row['cited_id']))
    

print('Graph fully connected:',nx.is_connected(graph))
print('Connected components:',nx.number_connected_components(graph))
giant_connected_component = list(max(nx.connected_components(graph), key=len))

# mask network by component
networks = networks[(networks['referring_id'].isin(giant_connected_component)) & (networks['cited_id'].isin(giant_connected_component))] # mask

# mask text by component
corpus = pd.DataFrame({'id':idx,'text':texts,'label':labels})
texts_new = corpus[corpus['id'].isin(giant_connected_component)]['text'].values.tolist()
idx_new = corpus[corpus['id'].isin(giant_connected_component)]['id'].values.tolist()
labels_new = corpus[corpus['id'].isin(giant_connected_component)]['label'].values.tolist()

# =============================================================================
# translate  node_indices to sequential numeric_indices
# =============================================================================
dictionary = dict()
idx_seq = list()
for i,value in tqdm(enumerate(idx_new)):
    dictionary[value]=i
    idx_seq.append(i)
networks_new = networks.replace(dictionary)

nodes_new = list(set(networks_new['referring_id'].values.tolist()+networks_new['cited_id'].values.tolist()))

if (len(nodes_new) == len(texts_new)) and (len(nodes_new) == len(giant_connected_component)):
    print('The dimensions are good now:',len(nodes_new))
else:
    print('Unmatching dimensions:', len(nodes_new) , len(texts_new) , len(giant_connected_component))

# =============================================================================
# Save for future use     - can skip
# =============================================================================
networks_new.to_csv(dir_root+'clean/single_component_small/network',index=False)
pd.DataFrame(texts_new).to_csv(dir_root+'clean/single_component_small/abstract_title all-lem',index=False,header=False)
pd.DataFrame(idx_new,columns=['id']).to_csv(dir_root+'clean/single_component_small/corpus_idx_original',index=False)
pd.DataFrame(labels_new,columns=['class1']).to_csv(dir_root+'clean/single_component_small/labels',index=False)
pd.DataFrame(idx_seq,columns=['id']).to_csv(dir_root+'clean/single_component_small/node_idx_seq',index=False)

# =============================================================================
# Prepare graph
# =============================================================================
graph = nx.Graph()
for i,row in tqdm(networks_new.iterrows(),total=networks_new.shape[0]):
    graph.add_edge(row['referring_id'],row['cited_id'])

print('Graph fully connected:',nx.is_connected(graph))
if nx.number_connected_components(graph)>1:
    print('Too many components in network. Should be 1, but is',nx.number_connected_components(graph))
else:
    print('Good. Connected components:',nx.number_connected_components(graph))

# connected_components = list(nx.connected_components(graph))
# del networks
gc.collect()
# =============================================================================
# Text embedding (BoW)
# =============================================================================
count_vect = CountVectorizer()
X = count_vect.fit_transform(texts_new)
X.shape
# pd.DataFrame(X).to_csv()
# =============================================================================
# Text embedding (TFIDF)
# =============================================================================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts_new)
X.shape
# =============================================================================
# TADW
# =============================================================================
TADW = kc.node_embedding.TADW(dimensions=32,lambd=1)
TADW.fit(graph,X)
TADW_vectors = TADW.get_embedding() 
pd.DataFrame(TADW_vectors).to_csv(dir_root+'embeddings/single_component_small/TADW-32-64-bow',index=False)
# =============================================================================
# TENE
# =============================================================================
TENE = kc.node_embedding.TENE(dimensions=32)
TENE.fit(graph,X)
TENE_vectors = TENE.get_embedding() 
pd.DataFrame(TENE_vectors).to_csv(dir_root+'embeddings/single_component_small/TENE-32-64-bow',index=False)
# =============================================================================
# DeepWalk
# =============================================================================
DW = kc.DeepWalk(dimensions=300)
DW.fit(graph)
DW_vectors = DW.get_embedding() 
pd.DataFrame(TENE_vectors).to_csv(dir_root+'embeddings/single_component_small/DW-300-240',index=False)
