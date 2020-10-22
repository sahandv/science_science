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
from DEC.DEC_keras import DEC_simple_run

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

results = pd.DataFrame([],columns=['Method','parameter','Silhouette','Homogeneity','NMI','AMI','ARI'])
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
print('\n- k-means random -----------------------')
for fold in tqdm(range(20)):
    seed = randint(0,10**5)
    model = KMeans(n_clusters=n_clusters,n_init=20, init='random', random_state=seed).fit(X)
    predicted_labels = model.labels_
    tmp_results = ['k-means random','seed '+str(seed)]+evaluate(X,Y,predicted_labels)
    tmp_results = pd.Series(tmp_results, index = results.columns)
    results = results.append(tmp_results, ignore_index=True)
mean = results.mean(axis=0)
maxx = results.max(axis=0)
print(mean)
print(maxx)
# =============================================================================
# K-means with init='k-means++'
# =============================================================================
print('\n- k-means++ -----------------------')
for fold in tqdm(range(20)):
    seed = randint(0,10**5)
    model = KMeans(n_clusters=n_clusters,n_init=20,init='k-means++', random_state=seed).fit(X)
    predicted_labels = model.labels_
    tmp_results = ['k-means++','seed '+str(seed)]+evaluate(X,Y,predicted_labels)
    tmp_results = pd.Series(tmp_results, index = results.columns)
    results = results.append(tmp_results, ignore_index=True)
mean = results.mean(axis=0)
maxx = results.max(axis=0)
print(mean)
print(maxx)
# =============================================================================
# Agglomerative
# =============================================================================
print('\n- Agglomerative -----------------------')
for fold in tqdm(range(4)):
    model = AgglomerativeClustering(n_clusters=n_clusters,linkage='ward').fit(X)
    predicted_labels = model.labels_
    tmp_results = ['Agglomerative','ward']+evaluate(X,Y,predicted_labels)
    tmp_results = pd.Series(tmp_results, index = results.columns)
    results = results.append(tmp_results, ignore_index=True)
mean = results.mean(axis=0)
maxx = results.max(axis=0)
print(mean)
print(maxx)
# =============================================================================
# DBSCAN
# =============================================================================
eps=0.000001
print('\n- DBSCAN -----------------------')
for fold in tqdm(range(19)):
    eps = eps+0.05
    model = DBSCAN(eps=eps, min_samples=10,n_jobs=15).fit(X)
    predicted_labels = model.labels_
    tmp_results = ['DBSCAN','eps '+str(eps)]+evaluate(X,Y,predicted_labels)
    tmp_results = pd.Series(tmp_results, index = results.columns)
    results = results.append(tmp_results, ignore_index=True)
mean = results.mean(axis=0)
maxx = results.max(axis=0)
print(mean)
print(maxx)
# =============================================================================
# Deep no min_max_scaling
# =============================================================================
archs = [[500, 500, 2000, 10],[500, 1000, 2000, 10],[500, 1000, 1000, 10],
         [500, 500, 2000, 100],[500, 1000, 2000, 100],[500, 1000, 1000, 100],
         [100, 300, 600, 10],[300, 500, 2000, 10],[700, 1000, 2000, 10],
         [200, 500, 10],[500, 1000, 10],[1000, 2000, 10],
         [200, 500, 100],[500, 1000, 100],[1000, 2000, 100],
         [1000, 500, 10],[500, 200, 10],[200, 100, 10],
         [1000, 1000, 2000, 10],[1000, 1500, 2000, 10],[1000, 1500, 1000, 10],
         [1000, 1000, 2000,500, 10],[1000, 1500, 2000,500, 10],[1000, 1500, 1000, 500, 10],
         [500, 500, 2000, 500, 10],[500, 1000, 2000, 500, 10],[500, 1000, 1000, 500, 10]]
print('\n- DEC -----------------------')
for fold in tqdm(archs):
    predicted_labels = DEC_simple_run(X,minmax_scale_custom_data=False,n_clusters=5,architecture=fold,pretrain_epochs=30)
    tmp_results = ['DEC',str(fold)]+evaluate(X,Y,predicted_labels)
    tmp_results = pd.Series(tmp_results, index = results.columns)
    results = results.append(tmp_results, ignore_index=True)
mean = results.mean(axis=0)
maxx = results.max(axis=0)
print(mean)
print(maxx)
# =============================================================================
# Deep with min_max_scaling
# =============================================================================
archs = [[500, 500, 2000, 10],[500, 1000, 2000, 10],[500, 1000, 1000, 10],
         [500, 500, 2000, 100],[500, 1000, 2000, 100],[500, 1000, 1000, 100],
         [100, 300, 600, 10],[300, 500, 2000, 10],[700, 1000, 2000, 10],
         [200, 500, 10],[500, 1000, 10],[1000, 2000, 10],
         [200, 500, 100],[500, 1000, 100],[1000, 2000, 100],
         [1000, 500, 10],[500, 200, 10],[200, 100, 10],
         [1000, 1000, 2000, 10],[1000, 1500, 2000, 10],[1000, 1500, 1000, 10],
         [1000, 1000, 2000,500, 10],[1000, 1500, 2000,500, 10],[1000, 1500, 1000, 500, 10],
         [500, 500, 2000, 500, 10],[500, 1000, 2000, 500, 10],[500, 1000, 1000, 500, 10]]
print('\n- DEC -----------------------')
for fold in tqdm(archs):
    predicted_labels = DEC_simple_run(X,minmax_scale_custom_data=True,n_clusters=5,architecture=fold,pretrain_epochs=30)
    tmp_results = ['DEC minmax scaler',str(fold)]+evaluate(X,Y,predicted_labels)
    tmp_results = pd.Series(tmp_results, index = results.columns)
    results = results.append(tmp_results, ignore_index=True)
mean = results.mean(axis=0)
maxx = results.max(axis=0)
print(mean)
print(maxx)
# =============================================================================
# 
# =============================================================================
