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
import plotly.express as px
from plotly.offline import plot
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
# from DEC.DEC_keras import DEC_simple_run
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
# Prepare
# =============================================================================
def plot_3D(X,labels,predictions,opacity=0.7):
    """
    Parameters
    ----------
    X : 2D np.array 
        Each row is expected to have values for X, Y, and Z dimensions.
    labels : iterable
        A 1D iterable (list or array) for labels of hover names.
    predictions : iterable
        A 1D iterable (list or array) for class labels. Must not be factorized.
    """
    X_df = pd.DataFrame(X_3d)
    X_df['class'] = predictions
    X_df.columns = ['x_ax','y_ax','z_ax','class']
    X_df = X_df.reset_index()
    X_df['labels'] = labels
    # X_grouped = X_df.groupby('class').groups
    
    fig = px.scatter_3d(X_df, x='x_ax', y='y_ax',z='z_ax', color='class', opacity=opacity,hover_name='labels') #.iloc[X_grouped[i]]
    plot(fig)
    
class CK_Means:
    """
        Initialization Parameters
        ----------
        k : int, optional
            The number of clusters for initial time slot. The default is 5.
        tol : float, optional
            Tolerance for centroid stability measure. The default is 0.00001.
        max_iter : int, optional
            Maximum number of iterations. The default is 300.
        patience : int, optional
            patience for telerance of stability. It is served as the minimum number of iterations after stability of centroids to continue the iters. The default is 2.
        boundary_thresh : float or None, optional
            Threshold to assign a node to a cluster out of the boundary of current nodes. The default is 0.5.
            If None, will automatically estimate.
        minimum_nodes : int, optional
            Minimum number of nodes to make a new cluster. The default is 10.
        seed : int, optional
            If seed is set, a seed will be used to make the results reproducable. The default is None.
        initializer : str, optional
            Centroid initialization. The options are:
                'random_generated' randomly generated centroid based on upper and lower bounds of data.  
                'random_selected' randomly selects an existing node as initialization point.
                The default is 'random_generated'.
        distance_metric : str, optional
            Options are 'euclidean' and 'cosine' distance metrics. The default is 'euclidean'.
        verbose : int, optional
            '1' outputs main iterations and steps
            '2' outputs debug data within each iteration, such as distances for tolerance
            The default is 1.
            
        Class Parameters
        ----------
        centroids_history: list
            List of centroids in each iteration. Can be used to plot the evolution.
        centoids: dict
            Current centroid values
        classifications: dict
            Vectors within each class. Can be used to visualize the classes.
        """
    def __init__(self, k:int=5, tol:float=0.00001, max_iter:int=300, patience=2, 
                 boundary_thresh:float=0.5, boundary_thresh_growth:float=1.1,minimum_nodes:int=10, seed=None,
                 initializer:str='random_generated',distance_metric:str='euclidean',
                 verbose:int=1):
        
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids_history = []
        self.centroids = {}
        self.seed = seed
        self.initializer = initializer
        self.distance_metric = distance_metric
        self.boundary_thresh = boundary_thresh
        self.minimum_nodes = minimum_nodes
        self.patience = patience
        self.v = verbose
        self.boundary_thresh_growth = boundary_thresh_growth
        if seed != None:
            np.random.seed(seed)
        
    def verbose(self,value=1,**outputs):
        if value<=self.v:
            for output in outputs.items():
                print('\n> '+str(output[0])+':',output[1])

    def initialize_rand_node_select(self,data):
        self.centroids = {}   
        for i in range(self.k):
            self.centroids[i] = data[i]    
            
    def initialize_rand_node_generate(self,data):
        self.centroids = {}   
        mat = np.matrix(X)
        self.golbal_boundaries = list(np.array([np.array(mat.max(0))[0],np.array(mat.min(0))[0]]).T)
        for i in range(self.k):
            self.centroids[i] = np.array([np.random.uniform(x[1],x[0]) for x in self.golbal_boundaries])
    
    def initialize_clusters(self):
        self.classifications = {}
        self.class_radius = {}
        for i in range(self.k):
            self.classifications[i] = []
            self.class_radius[i] = None
            
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
                self.verbose(2,debug=str(np.sum((current_centroid-original_centroid)/original_centroid*100.0))+' > '+str(self.tol))
                stable = False
        return stable
    
    def get_class_min_bounding_box(self,classifications):
        """
        Parameters
        ----------
        classifications : dict
            Dict of classifications provided by self.classifications

        Returns
        -------
        list
            list of minimum bounding boxes for each cluster/class.

        """
        self.ndbbox = []
        for i in classifications:
            i = np.array(classifications[i])
            try:
                self.ndbbox.append(np.array([i.min(axis=0,keepdims=True)[0],i.max(axis=0,keepdims=True)[0]]))
            except :
                self.ndbbox.append(np.zeros((2,i.shape[1])))
                self.verbose(2,warning='Class is empty! returning zero box.')
        return self.ndbbox

    def get_class_radius(self,classifications,centroids,distance_metric='euclidean'):
        """        
        Parameters
        ----------
        classifications : dict
            Dict of classifications provided by self.classifications
        centroids : dict
            Dict of centroids, provided by self.centroids.

        Returns
        -------
        list
            list of cluster/class radius.

        """
        self.radius = []
        for i in classifications:
            cluster = np.array(classifications[i])
            centroid = np.array(centroids[i])
            try:
                if distance_metric == 'euclidean':
                    self.radius.append(max([np.linalg.norm(vector-centroid) for vector in cluster]))
                if distance_metric == 'cosine':
                    self.radius.append(max([spatial.distance.cosine(vector,centroid) for vector in cluster]))
            except:
                self.radius.append(0)
        return self.radius
    
    def predict(self,data):
        assert len(data.shape)==2, "Incorrect shapes. Expecting a 2D np.array."
        labels = list()
        for featureset in data:
            if self.distance_metric=='cosine':
                distances = [spatial.distance.cosine(featureset,self.centroids[centroid]) for centroid in self.centroids]
            if self.distance_metric=='euclidean':
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            labels.append(distances.index(min(distances)))
        return labels


    def fit(self,data):
        """
        Perform clustering for T0

        Parameters
        ----------
        data : 2D numpy array.
            Array of feature arrays.

        """
        # Initialize centroids
        self.verbose(1,debug='Initializing centroids')
        patience_counter = 0
        if self.initializer=='random_generated':
            self.initialize_rand_node_generate(data)
        elif self.initializer=='random_selected':
            self.initialize_rand_node_select(data)
        
        # Iterations
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
            
            # Compare centroid change to stop iteration
            if self.centroid_stable():
                patience_counter+=1
                if patience_counter>self.patience:
                    self.verbose(1,debug='Centroids are stable within tolerance. Stopping.')
                    break
                self.verbose(2,debug='Centroids are stable within tolerance. Remaining patience:'+str(self.patience-patience_counter))
            else:
                patience_counter=0
                


    def fit_update(self,additional_data,previous_data):
        # Calculate cluster boundaries by finding min/max boundaries by np.matrix.min/max. (the simple way)
        self.verbose(1,debug='Initiating cluster distances and boundaries...')
        self.get_class_radius(self.classifications,self.centroids,self.distance_metric)
        self.model.get_class_min_bounding_box(self.classifications)

        self.update_assigned_labels = pd.DataFrame([],columns=[i for i in range(additional_data.shape[1])]+['label'])
        self.update_classes = []
        
        self.verbose(1,debug='Assigning ')
        for vec in tqdm(additional_data,total=additional_data.shape[0]):
            # If the new node is within boundary thresholds of any cluster, add to the cluster.
            #measure distance from centroids 
            if self.distance_metric=='cosine':
                distances = [spatial.distance.cosine(vec,self.centroids[centroid]) for centroid in self.centroids]
            if self.distance_metric=='euclidean':
                distances = [np.linalg.norm(vec-self.centroids[centroid]) for centroid in self.centroids]
            #nominate closest class
            distance = min(distances)
            classification = distances.index(distance)
            #get the radius of the class
            radius = self.radius[classification]
            
            # is it inside class or within class threshold?
            if distance <= radius+self.boundary_thresh:
                #yes: assign it
                self.classifications[classification].append(vec)
                # self.update_assigned_labels[vec].append(classification)
                self.update_assigned_labels = self.update_assigned_labels.append(list(vec)+[classification])
                # no: 
            else:
                # put it into a temprory new cluster and give it a name (K+1)
                self.classifications[model.k].append(vec)
                self.k+=1
                self.update_classes.append(self.k)
            
            # update centroids
            prev_centroids = dict(self.centroids)
            self.centroids_history.append(prev_centroids)
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
            # update radiuses 
            self.get_class_radius(self.classifications,self.centroids,self.distance_metric)
        
        # self.verbose(1,debug='Finalizing initial assignment by calculating final radius and boundaries.')
        # self.get_class_radius(self.classifications,self.centroids,self.distance_metric)
        # self.model.get_class_min_bounding_box(self.classifications)
        self.verbose(1,debug='Initial assignment completed.')
        
        # Clean the new classes up
        self.verbose(1,debug='Cleaning up now by removing low population classes...')
        #after the end of the loop, is cluster population<minimum_nodes?
        for ucl in self.update_classes:
            #yes: destroy cluster and assign nodes to nearby clusters.
            if len(self.classifications[ucl])<self.minimum_nodes:
                try:
                    self.verbose(2,debug='Popping and centroid class due to low population '+str(ucl))
                    self.classifications[ucl].pop(ucl)
                    self.centroids[ucl].pop(ucl)
                    self.update_classes.remove(ucl)
                except KeyError:
                    print('A KeyError detected. Unable to remove the class with class population under the threshold from self.classifications or self.centroids.')
                    sys.exit(1)
                    
                self.verbose(2,debug='Unassigning nodes from removed class '+str(ucl))
                self.update_assigned_labels[self.update_assigned_labels['label']==ucl]['label'] = None
        
        self.verbose(2,debug='Recording nodes with None labels, AKA orphaned nodes.')
        self.orphan_idx = self.update_assigned_labels[self.update_assigned_labels['label']==None].index
        self.orphan_vecs = self.update_assigned_labels[self.update_assigned_labels['label']==None].drop('label',axis=1)
            
        self.verbose(1,debug='Recalculating radius and boundaries for all remaining classes.')
        self.get_class_radius(self.classifications,self.centroids,self.distance_metric)
        self.model.get_class_min_bounding_box(self.classifications)
        
        
        self.verbose(1,debug='Reassigning orphaned nodes...')
        for orphan in self.orphan_vecs.iterrows():            
            if self.distance_metric=='cosine':
                distances = [spatial.distance.cosine(orphan.values,self.centroids[centroid]) for centroid in self.centroids]
            if self.distance_metric=='euclidean':
                distances = [np.linalg.norm(orphan.values-self.centroids[centroid]) for centroid in self.centroids]
            #nominate closest class
            distance = min(distances)
            if distance< radius+(self.boundary_thresh*self.boundary_thresh_growth):
                classification = distances.index(distance) # BUG: Distance index won't be the same as class index.
                #get the radius of the class
                radius = self.radius[classification]
                
                
                
        self.verbose(1,debug='Checking for class intersections by comparing centroid distances to sum of radiuses.')

        
    def fit_new(self,additional_data):
        
        


    def fit_legacy(self,data):
        # Initialize centroids
        self.initialize_rand_node_select(self,data)
        
        for i in tqdm(range(self.max_iter),total=self.max_iter):
            
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

X_0 = X[:-5000]
X_0.shape
Y_0 = Y[:-5000]
Y_0.shape

model = CK_Means(verbose=1,k=n_clusters,distance_metric='cosine')
model.fit(X_0)
predicted_labels = model.predict(X_0)

tmp_results = ['Ck-means T0','cosine']+evaluate(X_0,Y_0,predicted_labels)
tmp_results = pd.Series(tmp_results, index = results.columns)
results = results.append(tmp_results, ignore_index=True)

X_3d = TSNE(n_components=3, n_iter=500, verbose=2).fit_transform(X_0)
plot_3D(X_3d,labels[:-5000],predicted_labels)

X_1 = X[-5000:-2500]
Y_1 = Y[-5000:-2500]


            




