#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:10:03 2020

@author: github.com/sahandv
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
datapath = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/' #Ryzen
# datapath = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/' #C1314


# data_address =  datapath+"Corpus/cora-classify/cora/embeddings/single_component_small_18k/n2v 300-70-20 p1q05"#node2vec super-d2v-node 128-70-20 p1q025"
# label_address = datapath+"Corpus/cora-classify/cora/clean/single_component_small_18k/labels"

data_address =  datapath+"Corpus/KPRIS/embeddings/deflemm/Doc2Vec patent_wos corpus"
label_address =  datapath+"Corpus/KPRIS/labels"

vectors = pd.read_csv(data_address)#,header=None)
labels = pd.read_csv(label_address,names=['label'])
labels.columns = ['label']

try:
    vectors = vectors.drop('Unnamed: 0',axis=1)
    print('\nDroped index column. Now '+data_address+' has the shape of: ',vectors.shape)
except:
    print('\nVector shapes seem to be good:',vectors.shape)

labels_f = pd.factorize(labels.label)
X = vectors.values
Y = labels_f[0]
n_clusters = len(list(labels.groupby('label').groups.keys())) 

results = pd.DataFrame([],columns=['Method','parameter','Silhouette','Homogeneity','Completeness','NMI','AMI','ARI'])
# =============================================================================
# Evaluation method
# =============================================================================
def evaluate(X,Y,predicted_labels):
    
    df = pd.DataFrame(predicted_labels,columns=['label'])
    if len(df.groupby('label').groups)<2:
        return [0,0,0,0,0,0]
    
    try:
        sil = silhouette_score(X, predicted_labels, metric='euclidean')
    except:
        sil = 0
        
    return [sil,
            homogeneity_score(Y, predicted_labels),
            homogeneity_score(predicted_labels, Y),
            normalized_mutual_info_score(Y, predicted_labels),
            adjusted_mutual_info_score(Y, predicted_labels),
            adjusted_rand_score(Y, predicted_labels)]

# =============================================================================
# Evaluate if you already have results and skip clustering (i.e. LDA)
# =============================================================================
prediction_results_address = datapath+"Corpus/KPRIS/LDA Results/_5/dataset_topic_scores.csv"
predictions = pd.read_csv(prediction_results_address)['class'].values # If you don't want to cluster and already have resutls
tmp_results = ['LDA unigram','max_df 0.8']+evaluate(None,Y,predictions)
tmp_results = pd.Series(tmp_results, index = results.columns)
results = results.append(tmp_results, ignore_index=True)


# =============================================================================
# K-means
# =============================================================================
print('\n- k-means random -----------------------')
for fold in tqdm(range(5)):
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
# print('\n- k-means++ -----------------------')
# for fold in tqdm(range(20)):
#     seed = randint(0,10**5)
#     model = KMeans(n_clusters=n_clusters,n_init=20,init='k-means++', random_state=seed).fit(X)
#     predicted_labels = model.labels_
#     tmp_results = ['k-means++','seed '+str(seed)]+evaluate(X,Y,predicted_labels)
#     tmp_results = pd.Series(tmp_results, index = results.columns)
#     results = results.append(tmp_results, ignore_index=True)
# mean = results.mean(axis=0)
# maxx = results.max(axis=0)
# print(mean)
# print(maxx)
# =============================================================================
# Agglomerative
# =============================================================================
print('\n- Agglomerative -----------------------')
for fold in tqdm(range(1)):
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
# eps=0.000001
# print('\n- DBSCAN -----------------------')
# for fold in tqdm(range(19)):
#     eps = eps+0.05
#     model = DBSCAN(eps=eps, min_samples=10,n_jobs=15).fit(X)
#     predicted_labels = model.labels_
#     tmp_results = ['DBSCAN','eps '+str(eps)]+evaluate(X,Y,predicted_labels)
#     tmp_results = pd.Series(tmp_results, index = results.columns)
#     results = results.append(tmp_results, ignore_index=True)
# mean = results.mean(axis=0)
# maxx = results.max(axis=0)
# print(mean)
# print(maxx)
# =============================================================================
# Deep no min_max_scaling
# =============================================================================
archs = [[500, 500, 2000, 10],[500, 1000, 2000, 10],[500, 1000, 1000, 10],
         # [500, 500, 2000, 100],[500, 1000, 2000, 100],[500, 1000, 1000, 100],
         # [100, 300, 600, 10],[300, 500, 2000, 10],[700, 1000, 2000, 10],
         [200, 500, 10],[500, 1000, 10],[1000, 2000, 10],
         [200, 500, 100],[500, 1000, 100],[1000, 2000, 100],
         # [1000, 500, 10],[500, 200, 10],[200, 100, 10],
         # [1000, 1000, 2000, 10],[1000, 1500, 2000, 10],[1000, 1500, 1000, 10],
         # [1000, 1000, 2000,500, 10],[1000, 1500, 2000,500, 10],[1000, 1500, 1000, 500, 10],
         [500, 500, 2000, 500, 10],[500, 1000, 2000, 500, 10],[500, 1000, 1000, 500, 10]]
archs = [[200,500,10],[200,500,10],[200,500,10],[200,500,10],[200,500,10]]
print('\n- DEC -----------------------')
for fold in tqdm(archs):
    seed = randint(0,10**4)
    np.random.seed(seed)
    predicted_labels = DEC_simple_run(X,minmax_scale_custom_data=False,n_clusters=n_clusters,architecture=fold,pretrain_epochs=300)
    tmp_results = ['DEC',str(seed)+' '+str(fold)]+evaluate(X,Y,predicted_labels)
    tmp_results = pd.Series(tmp_results, index = results.columns)
    results = results.append(tmp_results, ignore_index=True)
mean = results.mean(axis=0)
maxx = results.max(axis=0)
print(mean)
print(maxx)
# =============================================================================
# Deep with min_max_scaling
# =============================================================================
# archs = [[500, 500, 2000, 10],[500, 1000, 2000, 10],[500, 1000, 1000, 10],
#          [500, 500, 2000, 100],[500, 1000, 2000, 100],[500, 1000, 1000, 100],
#          [100, 300, 600, 10],[300, 500, 2000, 10],[700, 1000, 2000, 10],
#          [200, 500, 10],[500, 1000, 10],[1000, 2000, 10],
#          [200, 500, 100],[500, 1000, 100],[1000, 2000, 100],
#          [1000, 500, 10],[500, 200, 10],[200, 100, 10],
#          [1000, 1000, 2000, 10],[1000, 1500, 2000, 10],[1000, 1500, 1000, 10],
#          [1000, 1000, 2000,500, 10],[1000, 1500, 2000,500, 10],[1000, 1500, 1000, 500, 10],
#          [500, 500, 2000, 500, 10],[500, 1000, 2000, 500, 10],[500, 1000, 1000, 500, 10]]
# print('\n- DEC -----------------------')
# for fold in tqdm(archs):
#     seed = randint(0,10**4)
#     np.random.seed(seed)
#     predicted_labels = DEC_simple_run(X,minmax_scale_custom_data=False,n_clusters=5,architecture=fold,pretrain_epochs=300)
#     tmp_results = ['DEC',str(seed)+' '+str(fold)]+evaluate(X,Y,predicted_labels)
#     tmp_results = pd.Series(tmp_results, index = results.columns)
#     results = results.append(tmp_results, ignore_index=True)
# mean = results.mean(axis=0)
# maxx = results.max(axis=0)
# print(mean)
# print(maxx)
# =============================================================================
# Just cluster
# =============================================================================

print('\n- k-means -----------------------')
seed = 11822
model = KMeans(n_clusters=n_clusters,n_init=20,init='k-means++', random_state=seed).fit(X)
predicted_labels = model.labels_

model = KMeans(n_clusters=n_clusters,n_init=20, init='random', random_state=seed).fit(X)
predicted_labels = model.labels_

results_df = pd.DataFrame(predicted_labels,columns=['label'])
results_df.to_csv(data_address+' Kmeans labels',index=False)

results_df.groupby('label').groups.keys()

archs = [[200,500,10],[200,500,10]]

print('\n- DEC -----------------------')
for i,fold in tqdm(enumerate(archs)):
    seed = randint(0,10**5)
    np.random.seed(seed)
    predicted_labels = DEC_simple_run(X,minmax_scale_custom_data=False,n_clusters=10,architecture=fold,pretrain_epochs=1000)
    tmp_results = ['DEC',str(seed)+' '+str(fold)]+evaluate(X,Y,predicted_labels)
    tmp_results = pd.Series(tmp_results, index = results.columns)
    results = results.append(tmp_results, ignore_index=True)
    
    results_df = pd.DataFrame(predicted_labels,columns=['label'])
    results_df.to_csv(data_address+' DEC 500, 1000, 1000, 500, 10 k10 labels - '+str(i),index=False)

results_df.groupby('label').groups.keys()

# =============================================================================
# Save to disk
# =============================================================================
results_df = pd.DataFrame(results)
results_df.to_csv(data_address+' clustering results - 06 2021',index=False)

# =============================================================================
# find centroid of each cluster for a model
# =============================================================================
cluster_centers = []
for cluster in tqdm(range(n_clusters),total=n_clusters):
    cluster_centers.append(np.array(X)[predicted_labels==cluster].mean(axis=0))
# =============================================================================
# Save clusters
# =============================================================================
predicted_labels = pd.DataFrame(predicted_labels,columns=['labels'])
predicted_labels.to_csv(data_address+' clustering predictions',index=False)