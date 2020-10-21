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
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, AffinityPropagation
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

results = pd.DataFrame([],columns=['Silhouette','Homogeneity','NMI','AMI','ARI'])
rand_states = []
method_names = []
# =============================================================================
# K-means
# =============================================================================
for fold in tqdm(range(20)):
    seed = randint(0,10**5)
    rand_states.append(seed)
    method_names.append('k-means')
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(X)
    predicted_labels = kmeans.labels_
    kmeans_results = [silhouette_score(X, predicted_labels, metric='euclidean'),
                    homogeneity_score(Y, predicted_labels),
                    normalized_mutual_info_score(Y, predicted_labels),
                    adjusted_mutual_info_score(Y, predicted_labels),
                    adjusted_rand_score(Y, predicted_labels)]
    kmeans_results = pd.Series(kmeans_results, index = results.columns)
    results = results.append(kmeans_results, ignore_index=True)
mean = results.mean(axis=0)
max = results.max(axis=0)

# =============================================================================
# Agglomerative
# =============================================================================



# =============================================================================
# Deep
# =============================================================================


