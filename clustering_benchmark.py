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
# Cluster and evaluate
# =============================================================================
def run_all_tests(data_address,output_file_name,labels,k):
    print('K=',k)
    print('Will write the results to',output_file_name)
    
    column_names = ['Method','parameter','Silhouette','Homogeneity','NMI','AMI','ARI']
    vectors = pd.read_csv(data_address)
    try:
        vectors = vectors.drop('Unnamed: 0',axis=1)
        print('\nDroped index column. Now '+data_address+' has the shape of: ',vectors.shape)
    except:
        print('\nVector shapes seem to be good:',vectors.shape)
        
    # data_dir+file_name+' dm_concat'
    labels_f = pd.factorize(labels.label)

    X = vectors.values
    Y = labels_f[0]
    n_clusters = k
    
    labels_task_1 = labels[(labels['label']=='car') | (labels['label']=='memory')]
    vectors_task_1 = vectors.iloc[labels_task_1.index]
    labels_task_1_f = pd.factorize(labels_task_1.label)
    X_task_1 = vectors_task_1.values
    Y_task_1 = labels_task_1_f[0]
    n_clusters_task_1 = 2
    
    results = pd.DataFrame([],columns=column_names)
    results_template = results.copy()
    # =============================================================================
    # Deep no min_max_scaling
    # =============================================================================
    archs = [
            [500, 500, 2000, 500],[500, 500, 2000, 500],[500, 500, 2000, 500],
            # [500, 500, 2000, 500],[500, 500, 2000, 500],[500, 500, 2000, 500],
            [500, 1000, 2000, 100],[500, 1000, 2000, 100],[500, 1000, 2000, 100],
            [200, 1000, 2000,100, 10],[200, 1000, 2000,200, 10],[200, 1000, 2000, 500, 10],
            # [200, 500, 1000, 500, 10],[200, 500, 1000, 200, 10],[200, 500, 1000, 100, 10],
            # [200, 1000, 2000,100, 10],[200, 1000, 2000,200, 10],[200, 1000, 2000, 500, 10],
            [200, 500, 1000, 500, 10],[200, 500, 1000, 200, 10],[200, 500, 1000, 100, 10],
            # [200, 1000, 2000,100],[200, 1000, 2000,200],[200, 1000, 2000, 500],
            # [200, 500, 1000, 500],[200, 500, 1000, 200],[200, 500, 1000, 100],
            # [200, 1000, 2000, 10],[200, 1000, 2000, 10],[200, 1000, 2000, 10],
            [1536,3072,1536,100],[1536,3072,1536,100],[1536,3072,1536,100],
            [256,1024,512,10],[256,1024,512,10],[256,1024,512,10],
            [1024,1024,2048,256,10],[1024,1024,2048,256,10],
            [512,1024,2048,128,10],[512,1024,2048,128,10],
            [512,1024,2048,10],[512,1024,2048,10],
            [1024,1024,2048,10],[1024,1024,2048,10],
            # [1536,768,384,192,10],[1536,768,384,192,10],[1536,768,192,10],
            [1536,3072,1536,100,10],[1536,3072,1536,100,10],[1536,3072,1536,100,10],
            # [1536,3072,1536,100],[1536,3072,1536,100],[1536,3072,1536,100],
            # [1536,3072,1536,10],[1536,3072,1536,10],[1536,3072,1536,10],
            # [1536,3072,1536,100,10],[1536,3072,1536,100,10],[1536,3072,1536,100,10],
            # [200,200,100],[200,200,100],[200,200,100],
            [200,500,20],[200,500,200],[200,500,200],
            # [200,200,10],[200,200,10],[200,200,10],
            # [400,400,10],[400,400,10],[400,400,10],
            # [400,400,10],[400,400,10],[400,400,10],
            [400,1000,10],[400,1000,10],[400,1000,100,10],[400,1000,100,10],
            [400,500,10],[400,500,10],[400,500,10],
            # [200,200,10],[200,200,10],[200,200,10],
            # [200,200,10],[200,200,10],[200,200,10],
            [200,200,10],[200,200,10],[200,200,10],
            # [200,500,10],[200,500,10],[200,500,10],
            # [200,500,10],[200,500,10],[200,500,10],
            # [200,500,10],[200,500,10],[200,500,10],
            # [200,500,10],[200,500,10],[200,500,10],
            [200,500,10],[200,500,10],[200,500,10]]
    print('\n- DEC -----------------------')
    for fold in tqdm(archs):
        gc.collect()
        seed = randint(0,10**4)
        np.random.seed(seed)
        try:
            predicted_labels = DEC_simple_run(X,minmax_scale_custom_data=False,n_clusters=n_clusters,architecture=fold,pretrain_epochs=1000)
            tmp_results = ['DEC',str(seed)+' '+str(fold)]+evaluate(X,Y,predicted_labels)
            tmp_results_s = pd.Series(tmp_results, index = results.columns)
            tmp_results_df = results_template.copy()
            tmp_results_df = tmp_results_df.append(tmp_results_s, ignore_index=True)
            results = results.append(tmp_results_s, ignore_index=True)
        except:
            print('Some error happened, skipping ',fold)
        
        print('writing the fold results to file')
        # if file does not exist write header 
        try:
            if not os.path.isfile(output_file_name):
                tmp_results_df.to_csv(output_file_name, header=column_names,index=False)
            else: # else it exists so append without writing the header
                tmp_results_df.to_csv(output_file_name, mode='a', header=False,index=False)
        except:
            print('something went wrong and could not write the results to file!\n',
                 'You may abort and see what can be done.\n',
                 'Or wait to see the the final results in memory.')
        
    mean = results.mean(axis=0)
    maxx = results.max(axis=0)
    print(mean)
    print(maxx)
    
    # # =============================================================================
    # # Deep with min_max_scaling
    # # =============================================================================
    # archs = [[500, 500, 2000, 10],[500, 1000, 2000, 10],[500, 1000, 1000, 10],
    #             [500, 500, 2000, 100],[500, 1000, 2000, 100],[500, 1000, 1000, 100],
    #             [100, 300, 600, 10],[300, 500, 2000, 10],[700, 1000, 2000, 10],
    #             [200, 500, 10],[500, 1000, 10],[1000, 2000, 10],
    #             [200, 500, 100],[500, 1000, 100],[1000, 2000, 100],
    #             [1000, 500, 10],[500, 200, 10],[200, 100, 10],
    #             [1000, 1000, 2000, 10],[1000, 1500, 2000, 10],[1000, 1500, 1000, 10],
    #             [1000, 1000, 2000,500, 10],[1000, 1500, 2000,500, 10],[1000, 1500, 1000, 500, 10],
    #             [500, 500, 2000, 500, 10],[500, 1000, 2000, 500, 10],[500, 1000, 1000, 500, 10],
    #             [200,200,10],[200,200,10],[200,200,10],
    #             [200,200,10],[200,200,10],[200,200,10],
    #             [200,200,10],[200,200,10],[200,200,10],
    #             [200,500,10],[200,500,10],[200,500,10],
    #             [200,500,10],[200,500,10],[200,500,10]]
    # print('\n- DEC 2-----------------------')
    # for fold in tqdm(archs):
    #     predicted_labels = DEC_simple_run(X,minmax_scale_custom_data=True,n_clusters=5,architecture=fold,pretrain_epochs=300)
    #     tmp_results = ['DEC minmax scaler',str(fold)]+evaluate(X,Y,predicted_labels)
    #     tmp_results = pd.Series(tmp_results, index = results.columns)
    #     results = results.append(tmp_results, ignore_index=True)
    # mean = results.mean(axis=0)
    # maxx = results.max(axis=0)
    # print(mean)
    # print(maxx)
    

    # =============================================================================
    # K-means
    # =============================================================================
    print('\n- k-means random -----------------------')
    for fold in tqdm(range(20)):
        seed = randint(0,10**5)
        model = KMeans(n_clusters=n_clusters,n_init=20, init='random', random_state=seed).fit(X)
        predicted_labels = model.labels_
        try:
            tmp_results = ['k-means random','seed '+str(seed)]+evaluate(X,Y,predicted_labels)
            tmp_results_s = pd.Series(tmp_results, index = results.columns)
            tmp_results_df = results_template.copy()
            tmp_results_df = tmp_results_df.append(tmp_results_s, ignore_index=True)
            results = results.append(tmp_results_s, ignore_index=True)
        except:
            print('Some error happened, skipping ',fold)
        
        print('writing the fold results to file')
        # if file does not exist write header 
        try:
            if not os.path.isfile(output_file_name):
                tmp_results_df.to_csv(output_file_name, header=column_names,index=False)
            else: # else it exists so append without writing the header
                tmp_results_df.to_csv(output_file_name, mode='a', header=False,index=False)
        except:
            print('something went wrong and could not write the results to file!\n',
                 'You may abort and see what can be done.\n',
                 'Or wait to see the the final results in memory.')
    mean = results.mean(axis=0)
    maxx = results.max(axis=0)
    print(mean)
    print(maxx)
    # =============================================================================
    # K-means with init='k-means++'
    # =============================================================================
    print('\n- k-means++ -----------------------')
    for fold in tqdm(range(20)):
        gc.collect()
        seed = randint(0,10**5)
        model = KMeans(n_clusters=n_clusters,n_init=20,init='k-means++', random_state=seed).fit(X)
        predicted_labels = model.labels_
        try:
            tmp_results = ['k-means++','seed '+str(seed)]+evaluate(X,Y,predicted_labels)
            tmp_results_s = pd.Series(tmp_results, index = results.columns)
            tmp_results_df = results_template.copy()
            tmp_results_df = tmp_results_df.append(tmp_results_s, ignore_index=True)
            results = results.append(tmp_results_s, ignore_index=True)
        except:
            print('Some error happened, skipping ',fold)
        
        print('writing the fold results to file')
        # if file does not exist write header 
        try:
            if not os.path.isfile(output_file_name):
                tmp_results_df.to_csv(output_file_name, header=column_names,index=False)
            else: # else it exists so append without writing the header
                tmp_results_df.to_csv(output_file_name, mode='a', header=False,index=False)
        except:
            print('something went wrong and could not write the results to file!\n',
                 'You may abort and see what can be done.\n',
                 'Or wait to see the the final results in memory.')
    mean = results.mean(axis=0)
    maxx = results.max(axis=0)
    print(mean)
    print(maxx)
    # =============================================================================
    # Agglomerative
    # =============================================================================
    print('\n- Agglomerative -----------------------')
    for fold in tqdm(range(4)):
        gc.collect()
        model = AgglomerativeClustering(n_clusters=n_clusters,linkage='ward').fit(X)
        predicted_labels = model.labels_
        try:
            tmp_results = ['Agglomerative','ward']+evaluate(X,Y,predicted_labels)
            tmp_results_s = pd.Series(tmp_results, index = results.columns)
            tmp_results_df = results_template.copy()
            tmp_results_df = tmp_results_df.append(tmp_results_s, ignore_index=True)
            results = results.append(tmp_results_s, ignore_index=True)
        except:
            print('Some error happened, skipping ',fold)
        
        print('writing the fold results to file')
        # if file does not exist write header 
        try:
            if not os.path.isfile(output_file_name):
                tmp_results_df.to_csv(output_file_name, header=column_names,index=False)
            else: # else it exists so append without writing the header
                tmp_results_df.to_csv(output_file_name, mode='a', header=False,index=False)
        except:
            print('something went wrong and could not write the results to file!\n',
                 'You may abort and see what can be done.\n',
                 'Or wait to see the the final results in memory.')
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
        gc.collect()
        eps = eps+0.05
        model = DBSCAN(eps=eps, min_samples=10,n_jobs=15).fit(X)
        predicted_labels = model.labels_
        try:
            tmp_results = ['DBSCAN','eps '+str(eps)]+evaluate(X,Y,predicted_labels)
            tmp_results_s = pd.Series(tmp_results, index = results.columns)
            tmp_results_df = results_template.copy()
            tmp_results_df = tmp_results_df.append(tmp_results_s, ignore_index=True)
            results = results.append(tmp_results_s, ignore_index=True)
        except:
            print('Some error happened, skipping ',fold)
        
        print('writing the fold results to file')
        # if file does not exist write header 
        try:
            if not os.path.isfile(output_file_name):
                tmp_results_df.to_csv(output_file_name, header=column_names,index=False)
            else: # else it exists so append without writing the header
                tmp_results_df.to_csv(output_file_name, mode='a', header=False,index=False)
        except:
            print('something went wrong and could not write the results to file!\n',
                 'You may abort and see what can be done.\n',
                 'Or wait to see the the final results in memory.')
    mean = results.mean(axis=0)
    maxx = results.max(axis=0)
    print(mean)
    print(maxx)

    # =============================================================================
    # Save to disk
    # =============================================================================
    # print('Writing to disk...')
    results_df = pd.DataFrame(results)
    # results_df.to_csv(output_file_name,index=False)
    print('Done.')
    return results_df
# =============================================================================
# Run
# =============================================================================
datapath = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/'

data_dir =  datapath+"Corpus/cora-classify/cora/"
label_address =  datapath+"Corpus/cora-classify/cora/corpus classes1 with citations"

vec_file_names = ['embeddings/Doc2Vec cora corpus dm1 with citations','embeddings/Doc2Vec cora_wos corpus dm1 with citations'#,'Doc2Vec patent corpus',
                  ,'embeddings/node2vec-80-10-128 p1q0.5','embeddings/node2vec-80-10-128 p4q1']
                  # 'FastText SIF patent_wos corpus','FastText SIF wos corpus']
labels = pd.read_csv(label_address,names=['label'])

clusters = labels.groupby('label').groups

for file_name in vec_file_names:
    gc.collect()
    output_file_name = data_dir+file_name+' clustering results'
    run_all_tests(data_dir+file_name,output_file_name,labels,len(list(clusters.keys())))

#%%
# =============================================================================
# Performance evaluation
# =============================================================================

import pandas as pd
# datapath = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/' #C1314
datapath = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/' #Zen
data_address =  datapath+"Corpus/cora-classify/cora/embeddings/node2vec-vanilla deepwalk 80-10-128 clustering results"
df = pd.read_csv(data_address)
max1 = df.groupby(['Method'], sort=False).max()
max2 = df.groupby(['Method']).agg({'NMI': 'max','AMI':'max','ARI':'max'})
max3 = df[df.groupby(['Method'])['NMI'].transform(max) == df['NMI']]
min3 = df[df.groupby(['Method'])['NMI'].transform(min) == df['NMI']]