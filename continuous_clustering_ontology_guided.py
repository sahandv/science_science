#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 08:33:08 2021

@author: github.com/sahandv
"""
import sys
import time
import gc
import os
import copy
import random
from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import plot
from random import randint
from scipy import spatial
import logging
import json
import itertools
from itertools import chain
from collections import Counter

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
from sklearn.neighbors import KernelDensity
import scipy.cluster.hierarchy as sch
from gensim.models import FastText as fasttext_gensim
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout,write_dot

from sciosci.assets import text_assets as ta
from sciosci.assets import advanced_assets as aa
# from DEC.DEC_keras import DEC_simple_run

pd.options.mode.chained_assignment = None  # default='warn'
logger = logging.getLogger(__name__)
tqdm.pandas()

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
            - The number of clusters for initial time slot. The default is 5.
        tol : float, optional
            - Tolerance for centroid stability measure. The default is 0.00001.
        n_iter : int, optional
            - Maximum number of iterations. The default is 300.
        patience : int, optional
            - patience for telerance of stability. It is served as the minimum 
            number of iterations after stability of centroids to continue the 
            iters. The default is 2.
        boundary_epsilon_coeff : float, optional, 
            - Coefficient of the boundary Epsilon, to calculate the exact epsilon 
            for each cluster. Used to assign a node to a cluster out of the 
            boundary of current nodes. 
            - Epsilon=Radius*boundary_epsilon_coeff
            - If zero, will not have evolutions.
            - The default is 0.1.
        boundary_epsilon_abs : float, optional
            **** DEPRECATED ****
            - Absolute value for boundary epsilon. 
            - If None, will ignore and use adaptive (boundary_epsilon_coeff) for calculations.
            - The default is None.
        minimum_nodes : int, optional
            - Minimum number of nodes to make a new cluster. The default is 10.
        a : float, optional, 
            - Weight or slope of temporal distance, while a>=0. 
            - The effectiveness of node in centroid calculation will be calculated 
            as in a weight function such as the default function ( V*[1/((a*t)+1)]), 
            where t is time delta, and V is vector value.
            - The default is 1.
        kernel_coeff : float, optional,
            - Density kernel is computed as the minimum radius value of the clusters. 
            The coefficient is multiplied with the computed value to yield the final 
            kernel bandwith as: bandwith=min(radius)*kernel_coeff
            - The default is 2.
        death_threshold : int, optional,
            - Classes with population below this amount will be elimined.
        growth_threshold_population: float, optional
            - Population in each class should grow over this amount to be considered a growing cluster, population-wise.
            - Default is 1.1
        growth_threshold_radius: float, optional
            - Radius in each class should grow over this amount to be considered a growing cluster, area-wise.
            - Default is 1.1
        seed : int, optional
            - If seed is set, a seed will be used to make the results reproducable. 
            - The default is None.
        initializer : str, optional
            - Centroid initialization. The options are:
                'random_generated' randomly generated centroid based on upper and lower bounds of data.  
                'random_selected' randomly selects an existing node as initialization point.
                The default is 'random_generated'.
        distance_metric : str, optional
            - Options are 'euclidean' and 'cosine' distance metrics. The default is 'euclidean'.
        verbose : int, optional
            - '1' verbosity outputs main iterations and steps
            - '2' verbosity outputs debug data within each iteration, such as distances for tolerance
            - The default is 1.
            
        Class Parameters
        ----------
        centroids_history: list
            List of centroids in each iteration. Can be used to plot the evolution.
        centoids: dict
            Current centroid values. Dict is used to preserve centroid id after classification eliminations/additions.
        classifications: Pandas DataFrame
            Comprises the vector data ('0','1',...,'n'), classification ('class'), and time slice ('t'). 
        """
    def __init__(self, k:int=5, tol:float=0.00001, n_iter:int=300, patience=2, 
                 boundary_epsilon_coeff:float=0.1, boundary_epsilon_abs:float=None, 
                 boundary_epsilon_growth:float=0, minimum_nodes:int=10, seed=None, 
                 a:float=1.0, kernel_coeff:float=2,death_threshold:int=10,
                 growth_threshold_population:float=1.1,growth_threshold_radius:float=1.1,
                 initializer:str='random_generated',distance_metric:str='euclidean',
                 verbose:int=1):
        
        self.k = k
        self.tol = tol
        self.n_iter = n_iter
        self.centroids_history = []
        self.centroids = {}
        self.seed = seed
        self.initializer = initializer
        self.distance_metric = distance_metric
        self.boundary_epsilon_coeff = boundary_epsilon_coeff
        self.minimum_nodes = minimum_nodes
        self.patience = patience
        self.a=a
        self.boundary_epsilon_coeff_growth = boundary_epsilon_growth
        self.kernel_coeff = kernel_coeff
        self.death_threshold = death_threshold
        self.growth_threshold_population = growth_threshold_population
        self.growth_threshold_radius = growth_threshold_radius
        self.evolution_events = {}
        self.evolution_event_counter = 0
        self.v = verbose
        
        if seed != None:
            np.random.seed(seed)
    
    def verbose(self,value,**outputs):
        """
        Verbose outputs

        Parameters
        ----------
        value : int
            An integer showing the verbose level of an output. Higher verbose value means less important.
            If the object's initialized verbose level is equal to or lower than this value, it will print.
        **outputs : kword arguments, 'key'='value'
            Outputs with keys. Keys will be also printed. More than one key and output pair can be supplied.
        """
        if value<=self.v:
            for output in outputs.items():
                print('\n> '+str(output[0])+':',output[1])
                
    def set_keyword_embedding_model(self,model):
        self.model_kw = model
    
    def set_ontology_tree(self,G):
        self.ontology_tree = G
    
    def set_ontology_keyword_search_index(self,index):
        self.ontology_search_index = index
    
    def set_ontology_dict(self,ontology):
        self.ontology_dict_base = ontology
    
    def vectorize_keyword(self,keyword):
        return np.array([self.model_kw.wv[key] for key in keyword.split(' ')]).mean(axis=0)
    
    def prepare_ontology(self):
        """
        Format the ontologies so each key will direct to its vector and level 2 parents efficiently

        """
        self.ontology_dict = {}
        for key in tqdm(self.ontology_dict_base):
            self.ontology_dict[key]={'parents':self.ontology_dict_base[key],'vector':self.vectorize_keyword(key)}    
        return self.ontology_dict
    
    def map_keyword_ontology(self,keyword):
        """
        Will return the most similar concept for a given keyword. Basically a search engine.
        """
        return list(self.ontology_dict.keys())[np.argmin(np.array([self.get_distance(self.vectorize_keyword(keyword),self.ontology_dict[i]['vector'],self.distance_metric) for i in self.ontology_dict]))]

    def map_keyword_ontology_from_index(self,keyword):
        """
        Will return the most similar concept for a given keyword from a pre-indexed search dictionary.
        """
        return self.ontology_search_index[keyword]


    def return_root_doc_vec(self,root:str,clssifications_portion,ignores:list):
        for i,row in clssifications_portion.iterrows():
            if i not in ignores:
                if root in row['roots'].values.tolist():
                    return row[self.columns_vector].values,i

    def graph_component_test(self,G,ratio_thresh:float=1/10,count_trhesh_low:int=10):
        if nx.number_connected_components(G)>1:
            self.verbose(2,debug=' -  -  - concept graph in class has '+str(nx.number_connected_components(G))+' connected components.')
            sub_graphs = list(G.subgraph(c) for c in nx.connected_components(G))
            edges_counts = [c.number_of_edges() for c in sub_graphs]
            edges_counts_l = [x for x in edges_counts if x>count_trhesh_low]
            to_split = [x for x in list(itertools.combinations(edges_counts_l, 2)) if min(x[0],x[1])/max(x[0],x[1])>ratio_thresh]
            self.verbose(2,debug=' -  -  - found splittable components: '+str(len(list(itertools.chain.from_iterable(to_split)))))
            to_split = list(itertools.chain.from_iterable(to_split))
            to_split_indices = [edges_counts.index(count) for count in to_split]
            concept_proposals = [list(sub_graphs[i].nodes())[0] for i in to_split_indices]
            return concept_proposals

    def initialize_rand_node_select(self,data):
        self.centroids = {}   
        for i in range(self.k):
            self.centroids[i] = data[i]
            
    def initialize_rand_node_generate(self,data):
        self.centroids = {}   
        mat = np.matrix(data)
        self.golbal_boundaries = list(np.array([np.array(mat.max(0))[0],np.array(mat.min(0))[0]]).T)
        for i in range(self.k):
            self.centroids[i] = np.array([np.random.uniform(x[1],x[0]) for x in self.golbal_boundaries])
    
    def initialize_clusters(self,data,keywords:list=None):
        """
        Make a Pandas DataFrame self.classifications from the 2D data, with empty class and T=0

        Parameters
        ----------
        data : 2D numpy array.
        keywords : list of lists

        """
        self.columns_vector = [str(i) for i in range(data.shape[1])]
        self.columns = ['t','class','kw']+self.columns_vector
        self.classifications = pd.DataFrame(data)
        self.classifications.insert(0,'class',None,0)
        self.classifications.insert(0,'t',None,0)
        self.classifications.insert(0,'kw',None,0)
        self.classifications.columns = self.columns
        self.classifications['kw'] = keywords
        self.classifications['t'] = 0
        self.class_radius = {}
        
        for i in range(self.k):
            self.class_radius[i] = None

            
    def get_distance(self,vec_a,vec_b,distance_metric:str='euclidean'):
        if distance_metric == 'euclidean':
            return np.linalg.norm(vec_a-vec_b)
        if distance_metric == 'cosine':
            return spatial.distance.cosine(vec_a,vec_b)
    
    def get_class_min_bounding_box(self,classifications):
        """
        Parameters
        ----------
        classifications: Pandas DataFrame
            Comprises the vector data ('0','1',...,'n'), classification ('class'), and time slice ('t').  provided by self.classifications

        Returns
        -------
        list
            list of minimum bounding boxes for each cluster/class.

        """
        self.ndbbox = {}
        labels = classifications.groupby('class').groups
        for i in labels:
            # i = np.array(classifications[i])
            vecs = classifications[classifications['class']==i][self.columns_vector].values
            try:
                self.ndbbox[i] = np.array([vecs.min(axis=0,keepdims=True)[0],vecs.max(axis=0,keepdims=True)[0]])
            except :
                self.ndbbox[i] = np.zeros((2,i.shape[1]))
                self.verbose(2,warning='Class is empty! returning zero box.')
        return self.ndbbox

    def get_epsilon_radius(self,epsilon_radius:float=None):
        if epsilon_radius==None:
            return min(self.radius.values())*self.boundary_epsilon_coeff
        else:
            return epsilon_radius

    def get_class_radius(self,classifications,centroids,distance_metric:str='euclidean',min_radius:float=None):
        """        
        Parameters
        ----------
        classifications: Pandas DataFrame
            Comprises the vector data ('0','1',...,'n'), classification ('class'), and time slice ('t').  provided by self.classifications
        centroids : dict
            Dict of centroids, provided by self.centroids.
        distance_metric : str, optional
        min_radius : flaot, optional
            A constant value for minimum radius of new-born clusters. (if applicable)
            If None, will automatically use the smallest radius epsilon from the available radius values.
            The default is None.
        Returns
        -------
        list
            list of cluster/class radius.

        """
        self.radius = {}
        labels = classifications.groupby('class').groups
        for i in labels:
            vecs = classifications[classifications['class']==i][self.columns_vector].values
            try:
                centroid = np.array(centroids[i])
                self.radius[i] = max([self.get_distance(vector,centroid,distance_metric) for vector in vecs])
            except:
                self.radius[i] = self.get_epsilon_radius(epsilon_radius=None)
                self.verbose(2,warning='Exception handled. During radius calculation for class '+str(i)+' an error occuured, so minimum radius was assigned for it.')
        return self.radius
    

    def add_to_clusters(self,data,t,keywords:list=None):
        """
        Update self.classifications using the new 2D data

        Parameters
        ----------
        data : 2D numpy array.
        t : int
            Time-stamp of data (e.g. 0,1,2,3,..,n)
        keywords: list of lists
        
        """
        classifications = pd.DataFrame(data)
        classifications.insert(0,'class',None,0)
        classifications.insert(0,'t',None,0)
        classifications.insert(0,'kw',None,0)
        classifications.columns = self.columns
        classifications['t'] = t
        classifications['kw'] = keywords
        
        self.classifications = self.classifications.append(classifications)
        self.classifications.reset_index(drop=True,inplace=True)
        
    def re_initialize_new_data_clusters(self,t):
        """
        Make a Pandas DataFrame self.classifications from the 2D data, with empty class and T=0
    
        Parameters
        ----------
        data : 2D numpy array.
        t: int
            Classes of time t will be re-set
        """
        self.classifications.loc[self.classifications['t']==t,'class'] = None

        # self.class_radius = {}
        # for i in range(self.k):
        #     self.class_radius[i] = None

    def centroid_stable(self):
        stable = True
        for c in self.centroids:
            try:
                original_centroid = self.centroids_history[-1][c] 
            except KeyError:
                self.verbose(2,debug="New classes and centroids added. considering it a movevement and returning False")
                stable = False
            current_centroid = self.centroids[c]
            movement = abs(np.sum((current_centroid-original_centroid)/abs(original_centroid)*100.0))
            if movement > self.tol:
                self.verbose(2,debug=str(movement)+' > '+str(self.tol))
                stable = False
            else:
                self.verbose(2,debug=str(movement)+' < '+str(self.tol))
        return stable
    
    def predict(self,data,distance_metric:str=None):
        assert len(data.shape)==2, "Incorrect shapes. Expecting a 2D np.array."
        if distance_metric==None:
            distance_metric = self.distance_metric
        labels = list()
        for featureset in data:
            distances = [self.get_distance(featureset,self.centroids[centroid],distance_metric) for centroid in self.centroids]
            labels.append(distances.index(min(distances)))
        return labels
    
    def weight(self,a,t):
        """
        Default weight function: W = 1/((a*time_delta)+1)
        a=-1 means the weight is zero. 

        """
        if a==-1:
            return 0
        
        return 1/((a*t)+1)

    def assign_cluster(self,vector,ignore:list):
        """
        Find the closes centroid to the vector and return the centroid index
    
        Parameters
        ----------
        vector : np.array
        ignore : list, optional
            list of classes to be ignored.
        Returns
        -------
        classification : int
    
        """
        if ignore==None:
            distances = [self.get_distance(vector,self.centroids[centroid],self.distance_metric) for centroid in self.centroids]
        else:
            distances = [self.get_distance(vector,self.centroids[centroid],self.distance_metric) for centroid in self.centroids if centroid not in ignore]
        classification = distances.index(min(distances)) #argmin: get the index of the closest centroid to this featureset/node
        return classification
    
    def assign_clusters(self,classifications,ignore:list=None):
        """
        Assign clusters to the list. Not recommended for using on the self.classifications dataframe, as may mix up the indices. If used, make sure to have the indices matched.
        Parameters
        ----------
        classifications: Pandas DataFrame
            Comprises the vector data ('0','1',...,'n'), classification ('class'), and time slice ('t'). 
    
        """
        # for i,featureset in enumerate(data):
        for i,row in classifications[self.columns_vector].iterrows():
            self.classifications['class'][i] = self.assign_cluster(row.values,ignore)
            
    def assign_clusters_pandas(self,t:int=None,ignore:list=None):
        """
    
        Parameters
        ----------
        classifications: Pandas DataFrame
            Comprises the vector data ('0','1',...,'n'), classification ('class'), and time slice ('t'). 
        t: int
            time slice to assign.
            Set to None to assign all
        ignore : list, optional
            list of classes to be ignored. Default is None.
        """
        if t==None:
            self.classifications['class'] = self.classifications[self.columns_vector].apply(lambda x: self.assign_cluster(x,ignore),axis = 1)
        else:
            self.classifications.loc[self.classifications['t']==t,'class'] = self.classifications[self.classifications['t']==t][self.columns_vector].apply(lambda x: self.assign_cluster(x,ignore),axis = 1)

    def cluster(self,t,n_iter,weights,ignore:list=None):
        """
        Parameters
        ----------
        t : int
            time slice.
        ignore : list, optional
            list of classes to be ignored. Default is None.

        Returns
        -------
        None.

        """
        patience_counter = 0
        for iteration in tqdm(range(n_iter),total=n_iter):
            # Re-initialize clusters
            self.re_initialize_new_data_clusters(t)
            
            # Assign clusters
            self.assign_clusters_pandas(t=t,ignore=ignore)
            
            # update centroids using time-dependant weighting scheme
            prev_centroids = dict(self.centroids)
            self.centroids_history.append(prev_centroids)
            for i in self.classifications.groupby('class').groups:
                vecs = self.classifications[self.classifications['class']==i][self.columns_vector].values
                vecs = (vecs.T*weights[self.classifications['class']==i].values.T).T
                self.centroids[i] = sum(vecs)/sum(weights[self.classifications['class']==i])
            
            # Compare centroid change to stop iteration
            if self.centroid_stable():
                patience_counter+=1
                if patience_counter>self.patience:
                    self.verbose(1,debug='Centroids are stable within tolerance. Stopping.')
                    break
                self.verbose(2,debug='Centroids are stable within tolerance. Remaining patience:'+str(self.patience-patience_counter))
            else:
                patience_counter=0

    def sub_cluster(self,t:int,to_split:int,new_centroids:list,n_iter:int,sub_k:int=2):
        """
        Parameters
        ----------
        t : int
            time slice.
        ignore : list, optional
            list of classes to be ignored. Default is None.
    
        Returns
        -------
        None.
    
        """
        patience_counter = 0
        self.verbose(2, debug='Getting class='+str(to_split)+' rows at t='+str(t))
        # select rows of the data to manipulate. index will be preserved so it can overwrite the correct rows.
        manipulation_classifcations = self.classifications[(self.classifications['class']==to_split) & (self.classifications['t']==t)]
        # get the list of all prior clusters.
        prev_clusters = list(dict(self.centroids).keys())
        # all prior clusters and current clusters are now in ignore list. So the subclust won't dedicate anythign to them.
        self.verbose(2, debug='Ignoring previous clusters and killing the old splitting cluster.')
        ignore = prev_clusters
        self.evolution_events[self.evolution_event_counter] = {'t':t,'c':to_split,'event':'death'}
        self.evolution_event_counter+=1
        # create new labels for the new proposed clusters
        new_cluster_ids = list(range(max(prev_clusters)+1,max(prev_clusters)+1+sub_k))
        self.verbose(2, debug='Initializeing centroids for new clusters: '+str(new_cluster_ids))
        # sub-sample the new centroids, so we cut the centrods to the desired numver of sub_k
        new_centroids = random.sample(new_centroids,sub_k)
        
        # initializing new centroids, hence new clusters
        for i,c in enumerate(new_cluster_ids):
            self.centroids[c] = new_centroids[i]
            self.evolution_events[self.evolution_event_counter] = {'t':t,'c':c,'event':'birth'}
            self.evolution_event_counter+=1
            
        self.verbose(2,debug='Starting the iterations...')
        for iteration in tqdm(range(n_iter),total=n_iter):
            # Re-initialize clusters
            # self.re_initialize_new_data_clusters(t)
            
            # Assign clusters
            self.assign_clusters(manipulation_classifcations,ignore=ignore)
            
            # update centroids using time-dependant weighting scheme
            prev_centroids = dict(self.centroids)
            self.centroids_history.append(prev_centroids)
            for i in new_cluster_ids:
                vecs = self.classifications[self.classifications['class']==i][self.columns_vector].values
                self.centroids[i] = np.average(vecs,axis=0)
            
            # Compare centroid change to stop iteration
            if self.centroid_stable():
                patience_counter+=1
                if patience_counter>self.patience:
                    self.verbose(1,debug='Centroids are stable within tolerance. Stopping sub-clustering for cluster '+str(to_split))
                    break
                self.verbose(2,debug='Centroids are stable within tolerance. Remaining patience:'+str(self.patience-patience_counter))
            else:
                patience_counter=0
                
    def fit(self,data,keywords:list=None):
        """
        Perform clustering for T0

        Parameters
        ----------
        data : 2D numpy array
            Array of feature arrays.

        """
        # Initialize centroids
        self.evolution_events[self.evolution_event_counter] = {'t':0,'c':None,'event':'birth'}
        self.evolution_event_counter+=1
        self.verbose(1,debug='Initializing centroids using method: '+self.initializer)
        
        if self.initializer=='random_generated':
            self.initialize_rand_node_generate(data)
        elif self.initializer=='random_selected':
            self.initialize_rand_node_select(data)
        self.verbose(1,debug='Initialized centroids')
        
        patience_counter = 0
        for iteration in tqdm(range(self.n_iter),total=self.n_iter):
            # Initialize clusters
            self.initialize_clusters(data,keywords)
            
            # Iterate over data rows and assign clusters
            self.assign_clusters_pandas()
                
            # Update centroids
            prev_centroids = dict(self.centroids)
            self.centroids_history.append(prev_centroids)
            for i in self.classifications.groupby('class').groups:
                vecs = self.classifications[self.classifications['class']==i][self.columns_vector].values
                self.centroids[i] = np.average(vecs,axis=0)
            
            # Compare centroid change to stop iteration
            if self.centroid_stable():
                patience_counter+=1
                if patience_counter>self.patience:
                    self.verbose(1,debug='Centroids are stable within tolerance. Stopping.')
                    break
                self.verbose(2,debug='Centroids are stable within tolerance. Remaining patience:'+str(self.patience-patience_counter))
            else:
                patience_counter=0
    
    
    def fit_update(self,additional_data,t,keywords:list=None,n_iter:int=None,weight=None,a:float=None):
        """
        Used for updating classifications/clusters for new data. Will use the previous data as weights for centroid handling. 
        You can set a to -1 to remove previous data effectiveness.

        Parameters
        ----------
        additional_data : 2D numpy array.
            Input data.
        t : int
            time sequence number.
        n_iter : int, optional
            Number of iterarions. The default is None.
        weight : TYPE, optional
            DESCRIPTION. The default is automatically calculated using a.
        a : float, optional
            Prior node weight slope. The default is 1.0.
        """
        
        initial_radius = self.get_class_radius(self.classifications,self.centroids,self.distance_metric)
        
        if n_iter==None:
            n_iter = self.n_iter
        if a==None:
            a = self.a        
        # self.update_assigned_labels = pd.DataFrame([],columns=[i for i in range(additional_data.shape[1])]+['label'])
        self.verbose(1,debug='Updating self.classifications with new data.')        
        self.add_to_clusters(additional_data,t,keywords)
        
        delta_t = abs(self.classifications['t']-self.classifications['t'].values.max())
        if weight==None:
            weights = self.weight(a,delta_t)
        else:
            try:
                weights = weight(a,delta_t)
            except:
                self.verbose(0,warning='Exception occuured while trying to get the weights. Please make sure to provide a valid weight function or use default by not providing anything. The function should accept two slope and delta_t inputs. Now will use the default one.')
                weights = self.weight(1,delta_t)
        # base_k = self.k
        
        # Cluster
        self.verbose(1,debug='Initial assignment...')
        self.cluster(t,n_iter,weights)

        # Modify clusters
        self.verbose(1,debug='Checking classes for death...')
        
        # cluster population check
        classifications_populations = self.classifications[self.classifications['t']==t][['class']].value_counts()
        classifications_populations = pd.DataFrame(classifications_populations,columns=['population'])
        classifications_populations.reset_index(inplace=True)
        
        to_ignore = []
        for i,row in classifications_populations.iterrows():
            if row['population'] < self.death_threshold:
                to_ignore.append(row['class'])
        
        self.classifications.loc[(self.classifications['t']==t) & (self.classifications['class'].isin(to_ignore)),'class'] = None
        if len(to_ignore) > 0:
            self.verbose(1,debug='Found dead clusters: '+str(to_ignore))
            self.verbose(1,debug='Clustering again with removal of the dead classes')
            self.evolution_events[self.evolution_event_counter] = {'t':t,'c':to_ignore,'event':'death'}
            self.evolution_event_counter +=1
            self.cluster(t,n_iter,weights,to_ignore)
            
        self.verbose(1,debug='Checking classes for death finalized.')
                
        # intera-cluster distances check
        self.verbose(2,debug='Checking classes for radius growth rates')
        new_radius = self.get_class_radius(self.classifications,self.centroids,self.distance_metric)
        for c in classifications_populations['class'].values.tolist(): 
            try:
                old_radius = initial_radius[c]
            except :
                self.verbose(2,debug=' - Class '+str(c)+' is new and not available in prior data.')
                continue
            if new_radius[c]/old_radius < 1:
                self.verbose(2,debug=' -  - Class '+str(c)+' is shrinking in radius.')
        
        
        self.verbose(1,debug='Checking classes for splitting...')
        # cluster population check
        self.verbose(2,debug=' - Checking classes for population proportion growth rates')
        classifications_populations_old = self.classifications[self.classifications['t']==t-1][['class']].value_counts() # we should consider to_ignore aka dead classes for correct total population  at t-1, not <t
        classifications_populations_old = pd.DataFrame(classifications_populations_old,columns=['population'])
        classifications_populations_old.reset_index(inplace=True)
        
        classifications_populations = self.classifications[self.classifications['t']==t][['class']].value_counts() # to_ignore is already eliminated in previous step, if any
        classifications_populations = pd.DataFrame(classifications_populations,columns=['population'])
        classifications_populations.reset_index(inplace=True)
        
        to_split = {}
        for c in classifications_populations['class'].values.tolist():
            try:
                class_population_ratio_old = classifications_populations_old[classifications_populations_old['class']==c]['population']/classifications_populations_old['population'].sum()
            except :
                self.verbose(2,debug=' -  - Class '+str(c)+' is new and not available in prior data.')
                continue
            class_population_ratio = classifications_populations[classifications_populations['class']==c]['population']/classifications_populations['population'].sum()
            growth_rate = class_population_ratio.tolist()[0]/class_population_ratio_old.tolist()[0]
            if growth_rate > self.growth_threshold_population:
                to_split[c] = 1
                self.verbose(2,debug=' -  - Class '+str(c)+' has grown in population more than the pre-defined threshold.')
        
        # intera-cluster distances check
        self.verbose(2,debug='Checking classes for radius growth rates')
        
        for c in classifications_populations['class'].values.tolist(): 
            try:
                old_radius = initial_radius[c]
            except :
                self.verbose(2,debug=' - Class '+str(c)+' is new and not available in prior data.')
                continue
            if new_radius[c]/old_radius > self.growth_threshold_radius:
                if c in to_split.keys():
                    to_split[c] +=1
                else:
                    to_split[c] = 1
                self.verbose(2,debug=' -  - Class '+str(c)+' has grown in radius more than the pre-defined threshold.')
                
        # density check
        # TBC
        
        # Ontology matching and checkin        
        self.verbose(2,debug=' - Checking alive classes at time t='+str(t)+' for splitting by ontologies.')
        self.verbose(2,debug=' -  - vectorizing concepts.')
        self.prepare_ontology()
        
        class_centroid_proposal = {}
        class_concepts = {}
        for c in classifications_populations['class'].values.tolist(): 
            self.verbose(2,debug=' -  - get records in this class.')
            clssifications_c = self.classifications[self.classifications['class']==c]
            
            self.verbose(2,debug=' -  - finding keywords in concepts.')
            try:
                self.verbose(2,debug=' -  - using search index instead of search engine.')
                clssifications_c['concepts'] = clssifications_c['kw'].progress_apply(lambda x: [self.map_keyword_ontology_from_index(key) for key in x])
            except:
                self.verbose(2,debug=' -  - using search engine. index not available. (It can be very slow!)')
                clssifications_c['concepts'] = clssifications_c['kw'].progress_apply(lambda x: [self.map_keyword_ontology(key) for key in x])
            
            
            self.verbose(2,debug=' -  - finding concept roots.')
            clssifications_c['roots'] = clssifications_c['kw'].progress_apply(lambda x: [self.ontology_dict[key]['parents'] for key in x])
            clssifications_c['roots'] = clssifications_c['roots'].progress_apply(lambda x: list(chain.from_iterable(x)))
            
            self.verbose(2,debug=' -  - generate document per concept ratios.')
            roots = clssifications_c['roots'].values.tolist()
            flat_roots = list(itertools.chain.from_iterable(roots))
            counts = pd.Index(flat_roots).value_counts()
            class_concepts[c] = counts
                        
            self.verbose(2,debug=' -  - finding classes with multiple concept roots and updating to_split list.')
            self.verbose(2,debug=' -  -  - generating concept graph in cluster')
            
            self.verbose(2,debug=' -  -  - preparing edges')
            roots = [r for r in roots if len(r)>1] # docs with at least two roots
            nodes = list(counts.keys())
            edges = list(itertools.chain.from_iterable([[list(set(list(x))) for x in list(itertools.combinations(sets, 2))] for sets in roots])) # make pairs, hence the links

            # edges_to_count = [[tuple(edge)] for edge in edges]
            # keys = list(Counter(itertools.chain(*edges_to_count)).keys())
            # vals = list(Counter(itertools.chain(*edges_to_count)).values())
            
            self.verbose(2,debug=' -  -  - constructing the graph')
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
            self.verbose(2,debug=' -  -  - checking for sub-graphs')
            concept_proposals = self.graph_component_test(G,ratio_thresh=1/self.growth_threshold_population,count_trhesh_low=self.death_threshold)
            
            if len(concept_proposals)>0:
                self.verbose(2,debug=' -  -  - cluster can be splitted for these concepts as centoids:'+str(concept_proposals))
                to_split[c] +=2
            else:
                self.verbose(2,debug=' -  -  - performing edge erosion level 1')
                to_delete = [list(x) for x in list(itertools.combinations(nodes, 2))]
                G.remove_edges_from(to_delete)
                self.verbose(2,debug=' -  -  - checking for sub-graphs after erosion level 1')
                concept_proposals = self.graph_component_test(G,ratio_thresh=1/self.growth_threshold_population,count_trhesh_low=self.death_threshold)
                
                if len(concept_proposals)>0:
                    self.verbose(2,debug=' -  -  - cluster can be splitted for these concepts as centoids:'+str(concept_proposals))
                    to_split[c] +=1
                else:
                    self.verbose(2,debug=' -  -  - performing edge erosion level 2')
                    to_delete = [list(x) for x in list(itertools.combinations(nodes, 2))]
                    G.remove_edges_from(to_delete)
                    self.verbose(2,debug=' -  -  - checking for sub-graphs after erosion level 2')
                    concept_proposals = self.graph_component_test(G,ratio_thresh=1/self.growth_threshold_population,count_trhesh_low=self.death_threshold)
                    
                    if len(concept_proposals)>0:
                        self.verbose(2,debug=' -  -  - cluster can be splitted for these concepts as centoids:'+str(concept_proposals))
                        to_split[c] +=0.5
                    else:
                        self.verbose(2,debug=' -  -  - cluster cannot be splitted')
            
            centroid_proposals = []
            ignores = [] #docs to ignore, as already selected
            self.verbose(2,debug=' -  - getting centroid proposals.')
            for root in list(concept_proposals):
                centroid_proposal,ignore = self.return_root_doc_vec(root,clssifications_c,ignores)
                centroid_proposals.append(centroid_proposal)
                ignores.append(ignore)
            class_centroid_proposal[c] = centroid_proposals
                    
        class_centroid_proposal = {k:v for k,v in class_centroid_proposal.items() if len(v)>=2}

        self.verbose(2,debug=' -  - sub-clustering the records in to_split classes.')
        
        while len(class_centroid_proposal)>0:
            print("Cluster split votes are as follows:")
            print(to_split)
            
            user_input = input("Which cluster you want to re-cluster? (N: none, A: Auto, or from: "+str([k for k,v in class_centroid_proposal.items()])+")\n")
            if user_input=='N':
                class_centroid_proposal = {}
            elif user_input=='A':
                for c,v in class_centroid_proposal.items():
                    if to_split[c]>=2:
                        self.vebose(1,debug=' -  -  sub clustering cluster '+str(c))
                        to_recluster = int(c)
                        centroid_vecs =  class_centroid_proposal[to_recluster]
                        self.sub_cluster(t, to_recluster, centroid_vecs, self.n_iter)
                        del class_centroid_proposal[to_recluster]
                class_centroid_proposal = {}    
            else:
                self.vebose(1,debug=' -  - sub clustering cluster '+str(user_input))
                to_recluster = int(user_input)
                centroid_vecs =  class_centroid_proposal[to_recluster]
                self.sub_cluster(t, to_recluster, centroid_vecs, self.n_iter)
                del class_centroid_proposal[to_recluster]
                
        
        self.verbose(1,debug='Checking classes for merging...')
        
        
    def cluster_neighborhood(self,to_inspect:list,to_ignore:list=None,epsilon:float=5.0):
        """
        Overlaps the bounding boxes to find the neighbouring clusters.

        Parameters
        ----------
        to_inspect : list
            list of wanted clusters.
        to_ignore : list of sets, optional
            List of unwanted cluster pairs. The default is None.
        epsilon : float, optional
            The epsilong to grow on each dimension for overlap creation. If>1.0, value will be used as percentage. The default is 0.05.

        Returns
        -------
        List of neighboring cluster as list of pairs.

        """
        pairs = list(itertools.combinations(to_inspect, 2))
        self.get_class_min_bounding_box(self.classifications)
        # self.ndbbox
        self.ndbbox_overlapping = {}
        for c in self.ndbbox.keys():
            if epsilon>1.0:
                epsilon_dim = (self.ndbbox[c][1,:]-self.ndbbox[c][0,:])*epsilon/100
                self.ndbbox_overlapping[c] = np.array([self.ndbbox[c][0,:]-epsilon_dim,self.ndbbox[c][1,:]+epsilon_dim])
            else:
                self.ndbbox_overlapping[c] = np.array([self.ndbbox[c][0,:]-epsilon,self.ndbbox[c][1,:]+epsilon])
        neighbours = []
        for pair in pairs:
            cond_a = (self.ndbbox_overlapping[pair[0]][0,:]<self.ndbbox_overlapping[pair[1]][1,:]).all()
            cond_b = (self.ndbbox_overlapping[pair[0]][1,:]>self.ndbbox_overlapping[pair[1]][0,:]).all()

            if cond_a and cond_b:
                neighbours.append(set(pair))
        neighbours = list(set([n for n in neighbours if n not in to_ignore]))
        return neighbours
        
        
#%% DIMENSIONS DATA        
# =============================================================================
# Load data and init
# =============================================================================
datapath = '/home/sahand/GoogleDrive/Data/' #Ryzen
# datapath = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/' #C1314

gensim_model_address = datapath+'Corpus/Dimensions All/models/Fasttext/gensim381/FastText100D-dim-scopus-update.model'
model_AI = fasttext_gensim.load(gensim_model_address)
# model_AI.save(datapath+'Corpus/Dimensions All/models/Fasttext/FastText100D-dim-scopus-update-gensim383.model')

with open(datapath+'Corpus/Taxonomy/concept_parents lvl2 - ontology_table') as f:
    ontology_table = json.load(f)



all_columns = pd.read_csv(datapath+'Corpus/Dimensions All/clean/data with abstract - tech and comp')[['FOR_initials','PY','id','FOR','DE-n']]
corpus_data = pd.read_csv(datapath+'Corpus/Dimensions All/clean/abstract_title method_b_3')
vectors = pd.read_csv(datapath+'Corpus/Dimensions All/embeddings/doc2vec 300D dm=1 window=10 b3')
vectors['id'] = corpus_data['id']
vectors = vectors[vectors['id'].isin(all_columns['id'])]
vectors = vectors.merge(all_columns, on='id', how='left')
# vectors.PY.hist(bins=60)
vectors['DE-n'] = vectors['DE-n'].str.split(';;;')
vectors['DE-n'] = vectors['DE-n'].progress_apply(lambda x: x[:5] if len(x)>4 else x) # cut off to 6 keywors only

# vectors.drop('id',axis=1,inplace=True)


# =============================================================================
# pre index data
# =============================================================================
k0 = 6
model = CK_Means(verbose=1,k=k0,distance_metric='cosine') 
model.v=3
model.set_ontology_dict(ontology_table)
model.set_keyword_embedding_model(model_AI)
ontology_dict = model.prepare_ontology()

all_keywords = list(set(list(itertools.chain.from_iterable(vectors['DE-n'].values.tolist()))))
all_vecs = [model.vectorize_keyword(k) for k in tqdm(all_keywords)]

del vectors
del corpus_data
del all_columns
del model_AI
gc.collect()

start = 17000*15
end = start+17000

distances = {}
for i,vec in tqdm(enumerate(all_vecs[start:end]),total=len(all_vecs[start:end])):
    distances[all_keywords[i]] = list(ontology_dict.keys())[np.argmin(np.array([spatial.distance.cosine(all_vecs[i],ontology_dict[o]['vector']) for o in ontology_dict]))]
    
output_address = 'Corpus/Dimensions All/clean/kw ontology search/'+str(start)+' keyword_search_pre-index.json'
with open(output_address, 'w') as json_file:
    json.dump(distances, json_file)


#%%



vectors_t0 = vectors[vectors.PY<2006]
keywords = vectors_t0['DE-n'].values.tolist()
vectors_t0.drop(['FOR_initials','PY','id','FOR','DE-n','id'],axis=1,inplace=True)

# dendrogram = aa.fancy_dendrogram(sch.linkage(vectors_t0, method='ward'),truncate_mode='lastp',p=800,show_contracted=True,figsize=(15,9)) #single #average #ward

model.fit(vectors_t0.values,keywords)
predicted_labels = model.predict(vectors_t0.values)
pd.DataFrame(predicted_labels).hist(bins=6)


model_backup = copy.deepcopy(model)
vectors_t1 = vectors[vectors.PY==2006]
keywords = vectors_t1['DE-n'].values.tolist()
vectors_t1.drop(['FOR_initials','PY','id','FOR','DE-n','id'],axis=1,inplace=True)


model_backup.fit_update(vectors_t1.values, 1, keywords)

# model_backup.get_class_radius(model.classifications,model.centroids,model.distance_metric)
# model_backup.get_class_radius(model.classifications,model.centroids,model.distance_metric)


#%% Play with ontology

def vectorize_keyword(keyword):
    return np.array([model_AI.wv[key] for key in keyword.split(' ')]).mean(axis=0)

def prepare_ontology(ontology):
    tmp = {}
    for key in tqdm(ontology):
        tmp[key]={'parents':ontology_table[key],'vector':vectorize_keyword(key)}    
    ontology = tmp
    return ontology
ontology_table = prepare_ontology(ontology_table)

def distance(vec_a,vec_b):
    return spatial.distance.cosine(vec_a, vec_b)

keyword = 'odor recognition system'
vector = vectorize_keyword(keyword)
nearest = list(ontology_table.keys())[np.argmin(np.array([distance(vector,ontology_table[i]['vector']) for i in tqdm(ontology_table)]))]
print(nearest,ontology_table[nearest])





#%% Tinker with objects

centroids = model.centroids
centroids_b = model_backup.centroids

classifications = model.classifications
classifications_b = model_backup.classifications

radius = model.get_class_radius(model.classifications,model.centroids,model.distance_metric)
radius_b = model_backup.get_class_radius(model_backup.classifications,model_backup.centroids,model_backup.distance_metric)

classifications_populations_old = model_backup.classifications[model_backup.classifications['t']==1-1][['class']].value_counts() # we should consider to_ignore aka dead classes for correct total population  at t-1, not <t
classifications_populations_old = pd.DataFrame(classifications_populations_old,columns=['population'])
classifications_populations_old.reset_index(inplace=True)

classifications_populations = model_backup.classifications[model_backup.classifications['t']==1][['class']].value_counts() # to_ignore is already eliminated in previous step, if any
classifications_populations = pd.DataFrame(classifications_populations,columns=['population'])
classifications_populations.reset_index(inplace=True)





#%% KPRIS DATA

# data_address =  datapath+"Corpus/cora-classify/cora/embeddings/single_component_small_18k/n2v 300-70-20 p1q05"#node2vec super-d2v-node 128-70-20 p1q025"
# label_address = datapath+"Corpus/cora-classify/cora/clean/single_component_small_18k/labels"

data_address =  datapath+"Corpus/KPRIS/embeddings/deflemm/Doc2Vec patent_wos corpus"
label_address =  datapath+"Corpus/KPRIS/labels"

vectors = pd.read_csv(data_address)#,header=None)
labels = pd.read_csv(label_address,names=['label'])
labels.columns = ['label']

try:
    vectors = vectors.drop('Unnamed: 0',axis=1)
    print('\nDroped index column. Now data has the shape of: ',vectors.shape)
except:
    print('\nVector shapes seem to be good:',vectors.shape)

data_address =  datapath+"Corpus/KPRIS/embeddings/deflemm/Doc2Vec patent_wos corpus"
labels_f = pd.factorize(labels.label)
vectors['labels'] = labels_f[0]
vectors_t0 = vectors[:5120]
vectors_t0 = vectors_t0.append(vectors[5120:9620])
vectors_t0 = vectors_t0.append(vectors[9720:14120])
vectors_t0 = vectors_t0.append(vectors[14220:15920])
vectors_t0 = vectors_t0.append(vectors[16920:17120])
vectors_t1 = vectors.drop(vectors_t0.index,axis=0)
vectors_t0 = vectors_t0.sample(frac=1).reset_index(drop=True)
vectors_t1 = vectors_t1.sample(frac=1).reset_index(drop=True)
print(vectors_t0.info())
print(vectors_t1.info())

Y = vectors['labels'].values
X = vectors.drop(['labels'],axis=1).values
Y_0 = vectors_t0['labels'].values
X_0 = vectors_t0.drop(['labels'],axis=1).values
Y_1 = vectors_t1['labels'].values
X_1 = vectors_t1.drop(['labels'],axis=1).values

n_clusters = len(list(labels.groupby('label').groups.keys())) 

results = pd.DataFrame([],columns=['Method','parameter','Silhouette','Homogeneity','Completeness','NMI','AMI','ARI'])


# =============================================================================
# Cluster 
# =============================================================================
print('\n- Custom clustering --------------------')
n_clusters = 4
print('k=',n_clusters)


for fold in range(1):
    np.random.seed(randint(0,10**5))
    model = CK_Means(verbose=1,k=n_clusters,distance_metric='cosine')
    model.fit(X_0)
    predicted_labels = model.predict(X_0)
    
    # start_time = time.time()
    # model.initialize_rand_node_generate(X_0)
    # print("--- %s seconds ---" % (time.time() - start_time))
    
    # model.initialize_clusters(X_0)
    # classifications = model.classifications
    # model.assign_clusters(X_0)
    
    tmp_results = ['Ck-means T0','cosine']+evaluate(X_0,Y_0,predicted_labels)
    tmp_results = pd.Series(tmp_results, index = results.columns)
    results = results.append(tmp_results, ignore_index=True)

classifications = model.classifications
centroids_history = model.centroids_history
centroids = model.centroids

model.kernel_coeff=2

n_samples = 100
model2 = copy.deepcopy(model)
# model2.add_to_clusters(X_1,1)
classifications2 = model2.classifications
# classifications2[classifications2['class']==None]
samples = list(classifications2[classifications2['t']==0].sample(n_samples).index)

x_0_samples = X_0[samples]
kde = KernelDensity(kernel='gaussian', bandwidth=2).fit(X_0)
scores_gaussian = kde.score_samples(x_0_samples)
scores_gaussian_f = np.exp(scores_gaussian)
# kde = KernelDensity(kernel='exponential', bandwidth=1.8).fit(X_0)
# scores_exponential = kde.score_samples(x_0_samples)

k_bandwith = 100 # number of nearest neighbours
neighbors = {}
distances = []
for sample_i in tqdm(samples):
    tmp = {}
    for sample_j in samples:
        distance = model.get_distance(X_0[sample_i],X_0[sample_j],'cosine')
        tmp[sample_j] = distance
        distances.append(distance)
    neighbors[sample_i] = pd.Series(tmp).nsmallest(k_bandwith)
percentile_value = np.percentile(np.array(distances),25)

# neighbors_b = neighbors.copy()
# for neighbor in tqdm(neighbors_b):
#     neighbors[neighbor] = neighbors[neighbor][neighbors[neighbor].values<percentile_value]
#     if len(neighbors[neighbor][neighbors[neighbor].values<percentile_value])<2:
#         del neighbors[neighbor]

diffs = {}
clusters_max = []
clusters_min = []
for neighbor in tqdm(neighbors):
    diffs[neighbor] = np.array([scores_gaussian[samples.index(neighbor)]-scores_gaussian[samples.index(j)] for j in list(neighbors[neighbor].index)])
    if (diffs[neighbor]>=-0.00001).all():
        clusters_max.append(neighbor)
    # if (diffs[neighbor]<=0.00001).all():
    #     clusters_min.append(neighbor)

model2.boundary_epsilon_coeff = 0.05
model2.v = 2
model2.fit_update(X_1,t=1,a=1)

# predicted_labels = model.predict(X_1)
# tmp_results = ['Ck-means T1','cosine']+evaluate(X_1,Y_1,predicted_labels)
# tmp_results = pd.Series(tmp_results, index = results.columns)
# results = results.append(tmp_results, ignore_index=True)
classifications2 = model2.classifications
classifications2[pd.isna(classifications2['class'])]
not_classified = classifications2[pd.isna(classifications2['class'])]




X_3d = TSNE(n_components=3, n_iter=500, verbose=2).fit_transform(X_0)
plot_3D(X_3d,labels[:-5000],predicted_labels)

X_1 = X[-5000:-2500]
Y_1 = Y[-5000:-2500]

model.get_class_radius(model.classifications,model.centroids,'cosine')

# plt.hist2d(X[:,0],X[:,1],bins=5)
# plt.show()


# from scipy.stats import kde
# import matplotlib.pyplot as plt
# x = np.random.normal(size=500)
# y = x * 3 + np.random.normal(size=500)
 
# # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
# nbins=300
# k = kde.gaussian_kde([x,y])
# xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
# zi = k(np.vstack([xi.flatten(), yi.flatten()]))
 
# # Make the plot
# plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
# plt.show()
 
# # Change color palette
# plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.Greens_r)
# plt.show()

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

print('\n- meanshift random -----------------------')
for fold in tqdm([0.22,0.45,0.9,1.2]):
    seed = randint(0,10**5)
    np.random.seed(seed)
    from sklearn.cluster import MeanShift
    model_ms = MeanShift(bandwidth=fold).fit(X)
    predicted_labels = model_ms.labels_
    tmp_results = ['mean shift','band='+str(fold)]+evaluate(X,Y,predicted_labels)
    tmp_results = pd.Series(tmp_results, index = results.columns)
    results = results.append(tmp_results, ignore_index=True)

np.unique(predicted_labels).shape
