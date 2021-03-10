#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:12:51 2020

@author: sahand

This is a preprocessing script for Cora dataset [McCallumIRJ]

@article{McCallumIRJ,
 author = "Andrew McCallum and Kamal Nigam and Jason Rennie and Kristie Seymore",
 title = "Automating the Construction of Internet Portals with Machine Learning",
 journal = "Information Retrieval Journal",
 volume = 3,
 pages = "127--163",
 publisher = "Kluwer",
 year = 2000,
 note = "www.research.whizbang.com/data"
}

"""
# =============================================================================
# Init
# =============================================================================
# dir_path = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/Corpus/cora-classify/cora/'      # ryzen
dir_path = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/cora-classify/cora/'      # c1314


import json   
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import gc
tqdm.pandas()
from sciosci.assets import text_assets

# =============================================================================
# read JSON and lists from Cora data
# =============================================================================
papers_list_raw = pd.read_csv(dir_path+'papers',sep='\t',names=['id','filename','citation string']) # contains duplicates
# papers_list_raw = papers_list_raw.groupby('id').first().reset_index()

papers_list_labeled = pd.read_csv(dir_path+'classifications',sep='\t',names=['filename','class'])
papers_list_labeled = papers_list_labeled[pd.notna(papers_list_labeled['class'])]


citations = pd.read_csv(dir_path+'citations',names=['referring_id','cited_id'],sep='\t')

# =============================================================================
# Prepare classes
# =============================================================================
def cleanup(arr):
    try:
        return np.array([x for x in arr if x!=''])
    except:
        print('\nGot',arr,', which is not a list. returning as-is.')
        return np.array(arr)
    
labels = pd.DataFrame(list(papers_list_labeled['class'].str.split('/').progress_apply(lambda x: cleanup(x))))
labels.columns = ['class1','class2','class3']
papers_list_labeled = pd.concat([papers_list_labeled,labels],axis=1)

# Inspect classes
label_names = [str(x) for x in list(labels.groupby('class1').groups.keys())]

# =============================================================================
# Read text files
# =============================================================================
mypath = dir_path+'extractions'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
columns = ['filename','URL','Refering-URL','Root-URL','Email','Title','Author','Address','Affiliation','Note','Abstract','References-found']
papers_df = pd.DataFrame([],columns=columns)
log = []

for f_name in tqdm(files): 
    # f_name = 'http:##www.win.tue.nl#win#cs#fm#Dennis.Dams#Papers#dgg95a.ps.gz'
    f = open(join(mypath,f_name), "r")
    paper = [['filename',f_name]]
    try:
        tmp = f.read().split('\n')
    except:
        print('Failed to read file ',f_name,'\nLook at the final log for the list of such files.')
        log.append(['reading failed',f_name])
        continue
    
    for line in tmp:
        if line!='':
            ar = line.split(': ', 1)
            if len(ar)>1:
                paper.append(ar)
                
    paper_np = np.array(paper)
    paper = pd.DataFrame(paper_np.T[1])
    paper.index = paper_np.T[0]
    paper = paper.T
    paper = paper[paper.columns[paper.columns.isin(columns)]]
    # papers_df = papers_df.append(paper)[papers_df.columns]
    try:
        papers_df = pd.concat([papers_df,paper])
    except:
        print('Something went wrong when concatenating the file',f_name,'\nLook at the final log for the list of such files.')
        log.append(['concatenating failed',f_name])
        
papers_df.to_csv(dir_path+'extractions.csv',index=False)
log=pd.DataFrame(log,columns=['error','file'])
log.to_csv(dir_path+'extractions_log')

# =============================================================================
# Merge based on file name to get the idx
# =============================================================================

merged = pd.merge(papers_df, papers_list_raw, on='filename')
merged.to_csv(dir_path+'extractions_with_id.csv',index=False)
sample = merged.sample(5)

# =============================================================================
# Further pre-process to get unique abstracts
# =============================================================================
data = pd.read_csv(dir_path+'extractions_with_id.csv')

# =============================================================================
# Merge based on file name to get the idx
# =============================================================================

merged = pd.merge(papers_list_labeled, data, on='filename')
merged.to_csv(dir_path+'extractions_with_unique_id_labeled.csv',index=False)
sample = merged.sample(5)

data = merged.copy()
# =============================================================================
# Save to disk
# =============================================================================
data = pd.read_csv(dir_path+'extractions_with_id.csv')

data_clean = data[pd.notna(data['Abstract'])]
data_clean = data_clean[data_clean['Abstract']!='']
data_clean = data_clean[data_clean['Abstract']!=' ']
data_clean = data_clean[pd.notnull(data_clean['Abstract'])]
data_clean = data_clean[pd.notna(data_clean['Title'])]
data_clean = data_clean[pd.notna(data_clean['id'])]
data_clean_unique = data_clean.groupby('id').first().reset_index()
data_clean_unique.to_csv(dir_path+'extractions_with_unique_id.csv',index=False)
sample = data_clean_unique.sample(500)

# =============================================================================
# Filter citations based on the papers
# =============================================================================
data = pd.read_csv(dir_path+'extractions_with_unique_id_labeled.csv')
sample = data.sample(500)
id_list = data['id'].values.tolist()

filtered_citations = citations[(citations['referring_id'].isin(id_list)) | (citations['cited_id'].isin(id_list))]
filtered_citations.to_csv(dir_path+'citations_filtered.csv',index=False)

# =============================================================================
# Filter and take the largest component
# =============================================================================

# dir_path = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/Corpus/cora-classify/cora/'
data = pd.read_csv(dir_path+'extractions_with_unique_id_labeled.csv')
citations = pd.read_csv(dir_path+'citations_filtered.csv')# with_d2v300D_supernodes.csv')#, names=['referring_id','cited_id'],sep='\t')
citations.columns = ['referring_id','cited_id']

citations.info(memory_usage='deep')

graph = nx.Graph()
for i,row in tqdm(citations.iterrows(),total=citations.shape[0]):
    graph.add_edge(row['referring_id'],(row['cited_id']))
    

print('Graph fully connected:',nx.is_connected(graph))
print('Connected components:',nx.number_connected_components(graph))
giant_connected_component = list(max(nx.connected_components(graph), key=len))

data_filtered = data[data['id'].isin(giant_connected_component)]
data_filtered.to_csv(dir_path+'extractions_with_unique_id_labeled_single_component.csv',index=False)


