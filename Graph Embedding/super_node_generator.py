#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:30:13 2021

@author: sahand
"""

import pandas as pd
datapath = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/'
idx_address = datapath+"Corpus/cora-classify/cora/clean/single_component_small_18k/node_idx_seq cocite"
cluster_address = datapath+"Corpus/cora-classify/cora/embeddings/single_component_small_18k/doc2vec 300D dm=1 window=10 clustering predictions"
citations_address = datapath+'Corpus/cora-classify/cora/citations'
citations_filter_address =  datapath+"Corpus/cora-classify/cora/clean/single_component_small_18k/corpus_idx_original"


idx = pd.read_csv(idx_address)
clusters = pd.read_csv(cluster_address)
citation_pairs = pd.read_csv(citations_address,sep='\t',names=['a','b'])

#filter -- optional
citations_filter =  pd.read_csv(citations_filter_address).values[:,0].tolist()
citation_pairs = citation_pairs[(citation_pairs['a'].isin(citations_filter)) | (citation_pairs['b'].isin(citations_filter))] # mask


citation_pairs = citation_pairs.values.tolist()

idx_clusters = idx.copy()
idx_clusters['cluster'] = clusters['labels']
cluster_ids = list(clusters.groupby('labels').groups.keys())
super_id = 50000000  # A number out of document ID ranges

# Connect the supernode to all cluster members
for cluster in cluster_ids:
    cluster_papers = idx_clusters[idx_clusters['cluster']==cluster]
    cluster_papers = cluster_papers.values
    cluster_papers[:,1] = cluster_papers[:,1]+super_id # ID for supernodes which wouldn't conflict with document IDs
    cluster_papers = cluster_papers.tolist()
    citation_pairs = citation_pairs+cluster_papers
    

citation_pairs = pd.DataFrame(citation_pairs,columns=['referring_id','cited_id'])#,'count'])
citation_pairs.to_csv(datapath+'Corpus/cora-classify/cora/clean/single_component_small_18k/network_cocitation with_d2v300D_supernodes',index=False)
