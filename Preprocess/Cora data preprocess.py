#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:12:51 2020

@author: sahand

This is a preprocessing script for Cora dataset*
* 

"""
# =============================================================================
# Init
# =============================================================================
dir_path = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/Corpus/cora-classify/cora/'

import json   
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
tqdm.pandas()
from sciosci.assets import text_assets

# =============================================================================
# read JSON and lists from Cora data
# =============================================================================
papers_list_raw = pd.read_csv(dir_path+'papers',sep='\t',names=['id','filename','citation string']) # contains duplicates
# papers_list_raw = papers_list_raw.groupby('id').first().reset_index()

papers_list = pd.read_csv(dir_path+'classifications',sep='\t',names=['paper_name','class'])

citations = pd.read_csv(dir_path+'citations',names=['referring_id','cited_id'],sep='\t')

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
data_clean = data[pd.notnull(data['Abstract'])]
data_clean = data_clean[pd.notna(data_clean['Title'])]
data_clean = data_clean[pd.notna(data_clean['id'])]
data_clean_unique = data_clean.groupby('id').first()
data_clean_unique.to_csv(dir_path+'extractions_with_unique_id.csv',index=False)
sample = data.sample(5)
