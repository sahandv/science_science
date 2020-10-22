#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:10:03 2020

@author: sahand
"""
import sys
import time
import gc
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from random import randint

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn import metrics
from sklearn.metrics.cluster import silhouette_score,homogeneity_score,adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score,adjusted_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfTransformer , TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sciosci.assets import text_assets as ta

# =============================================================================
# Load data and init
# =============================================================================
datapath = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/'

data_address =  datapath+"Corpus/KPRIS/embeddings/deflemm/Doc2Vec patent corpus"
label_address =  datapath+"Corpus/KPRIS/labels"

vectors = pd.read_csv(data_address)
labels = pd.read_csv(label_address,names=['label'])
labels_f = pd.factorize(labels.label)
X = vectors.values
Y = labels_f[0]
n_clusters = 5

labels_task_1 = labels[(labels['label']=='car') | (labels['label']=='memory')]
vectors_task_1 = vectors.iloc[labels_task_1.index]
labels_task_1_f = pd.factorize(labels_task_1.label)
X_task_1 = vectors_task_1.values
Y_task_1 = labels_task_1_f[0]
n_clusters_task_1 = 2

results = pd.DataFrame([],columns=['Method','Silhouette','Homogeneity','NMI','AMI','ARI'])
rand_states = []
method_names = []
# =============================================================================
# Evaluation method
# =============================================================================
def evaluate(X,Y,predicted_labels):
    
    df = pd.DataFrame(predicted_labels,columns=['label'])
    if len(df.groupby('label').groups)<2:
        return [0,0,0,0,0]
    
    return [silhouette_score(X, predicted_labels, metric='euclidean'),
                    homogeneity_score(Y, predicted_labels),
                    normalized_mutual_info_score(Y, predicted_labels),
                    adjusted_mutual_info_score(Y, predicted_labels),
                    adjusted_rand_score(Y, predicted_labels)]
# =============================================================================
# K-means
# =============================================================================
for fold in tqdm(range(20)):
    seed = randint(0,10**5)
    rand_states.append(seed)
    model = KMeans(n_clusters=n_clusters, random_state=seed).fit(X)
    predicted_labels = model.labels_
    tmp_results = ['k-means']+evaluate(X,Y,predicted_labels)
    tmp_results = pd.Series(tmp_results, index = results.columns)
    results = results.append(tmp_results, ignore_index=True)
mean = results.mean(axis=0)
maxx = results.max(axis=0)
print('k-means')
print(mean)
print(maxx)
# =============================================================================
# K-means with init='k-means++'
# =============================================================================
for fold in tqdm(range(20)):
    seed = randint(0,10**5)
    rand_states.append(seed)
    model = KMeans(n_clusters=n_clusters,init='k-means++', random_state=seed).fit(X)
    predicted_labels = model.labels_
    tmp_results = ['k-means++']+evaluate(X,Y,predicted_labels)
    tmp_results = pd.Series(tmp_results, index = results.columns)
    results = results.append(tmp_results, ignore_index=True)
mean = results.mean(axis=0)
maxx = results.max(axis=0)
print('k-means')
print(mean)
print(maxx)
# =============================================================================
# Agglomerative
# =============================================================================
for fold in tqdm(range(4)):
    model = AgglomerativeClustering(n_clusters=n_clusters,linkage='ward').fit(X)
    predicted_labels = model.labels_
    tmp_results = ['Agglomerative']+evaluate(X,Y,predicted_labels)
    tmp_results = pd.Series(tmp_results, index = results.columns)
    results = results.append(tmp_results, ignore_index=True)
mean = results.mean(axis=0)
maxx = results.max(axis=0)
print('Agglomerative')
print(mean)
print(maxx)
# =============================================================================
# DBSCAN
# =============================================================================
eps=0.000001
for fold in tqdm(range(19)):
    eps = eps+0.05
    model = DBSCAN(eps=eps, min_samples=10,n_jobs=15).fit(X)
    predicted_labels = model.labels_
    tmp_results = ['DBSCAN']+evaluate(X,Y,predicted_labels)
    tmp_results = pd.Series(tmp_results, index = results.columns)
    results = results.append(tmp_results, ignore_index=True)
mean = results.mean(axis=0)
maxx = results.max(axis=0)
print('DBSCAN')
print(mean)
print(maxx)
# =============================================================================
# Deep
# =============================================================================


