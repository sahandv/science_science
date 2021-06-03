#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 10:20:31 2021

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
from scipy import spatial

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
# datapath = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/' #Ryzen
datapath = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/' #C1314


data_address =  datapath+"Corpus/cora-classify/cora/embeddings/single_component_small_18k/n2v 300-70-20 p1q05"#node2vec super-d2v-node 128-70-20 p1q025"
label_address = datapath+"Corpus/cora-classify/cora/clean/single_component_small_18k/labels"

# data_address =  datapath+"Corpus/KPRIS/embeddings/deflemm/Doc2Vec patent_wos corpus"
# label_address =  datapath+"Corpus/KPRIS/labels"

vectors = pd.read_csv(data_address)#,header=None)
labels = pd.read_csv(label_address)
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

results = pd.DataFrame([],columns=['Method','parameter','Silhouette','Homogeneity','NMI','AMI','ARI'])
# =============================================================================
# Evaluation method
# =============================================================================
def evaluate(X,Y,predicted_labels):
    
    df = pd.DataFrame(predicted_labels,columns=['label'])
    if len(df.groupby('label').groups)<2:
        return [0,0,0,0,0]
    
    try:
        sil = silhouette_score(X, predicted_labels, metric='euclidean')
    except:
        sil = 0
        
    return [sil,
            homogeneity_score(Y, predicted_labels),
            normalized_mutual_info_score(Y, predicted_labels),
            adjusted_mutual_info_score(Y, predicted_labels),
            adjusted_rand_score(Y, predicted_labels)]

# =============================================================================
# Cluster benchmark
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
# Cluster 
# =============================================================================
print('\n- Custom clustering --------------------')
print(n_clusters)



class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300,seed=None,initializer='random_generated',distance_metric='euclidean'):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids_history = []
        self.centroids = {}
        self.seed = seed
        self.initializer = initializer
        self.distance_metric = distance_metric
        if seed != None:
            np.random.seed(seed)
        
    def initialize_rand_node_select(self,data):
        self.centroids = {}   
        for i in range(self.k):
            self.centroids[i] = data[i]    
            
    def initialize_rand_node_generate(self,data):
        self.centroids = {}   
        Mat = np.matrix(X)
        self.boundaries = list(np.array([np.array(Mat.max(0))[0],np.array(Mat.min(0))[0]]).T)
        for i in range(self.k):
            self.centroids[i] = np.array([np.random.uniform(x[1],x[0]) for x in self.boundaries])
    
    def initialize_clusters(self):
        self.classifications = {}
        for i in range(self.k):
            self.classifications[i] = []
    
    def assign_clusters(self,data):
        for featureset in data:
            if self.distance_metric=='cosine':
                distances = [spatial.distance.cosine(featureset,self.centroids[centroid]) for centroid in self.centroids]
            if self.distance_metric=='euclidean':
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances)) #argmin: get the index of the closest centroid to this featureset/node
            self.classifications[classification].append(featureset)
    
    def centroid_stable(self):
        stable = True
        for c in self.centroids:
            original_centroid = self.centroids_history[-1][c] 
            current_centroid = self.centroids[c]
            if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                stable = False
        return stable
    
    def fit(self,data):
        # Initialize centroids
        print('Initializing centroids')
        if self.initializer=='random_generated':
            self.initialize_rand_node_generate(data)
        elif self.initializer=='random_selected':
            self.initialize_rand_node_select(data)
        
        # Iterate over data
        for iteration in tqdm(range(self.max_iter),total=self.max_iter):
            
            # Initialize clusters
            self.initialize_clusters()
            
            # Iterate over data rows and assign clusters
            self.assign_clusters(data)
                
            # Update centroids
            prev_centroids = dict(self.centroids)
            self.centroids_history.append(prev_centroids)
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
            
            # Compare change to stop iteration
            if self.centroid_stable():
                break

    def fit_update(self,data):
        pass

    def predict(self,featureset):
        if self.distance_metric=='cosine':
            distances = [spatial.distance.cosine(featureset,self.centroids[centroid]) for centroid in self.centroids]
        if self.distance_metric=='euclidean':
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


    def fit_legacy(self,data):
        # Initialize centroids
        self.initialize_rand_node_select(self,data)
        
        for i in range(self.max_iter):
            
            # Initialize clusters
            self.classifications = {}
            for i in range(self.k):
                self.classifications[i] = []
                
            # Iterate over data rows and assign clusters
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            # Update centroids
            prev_centroids = dict(self.centroids)
            
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

