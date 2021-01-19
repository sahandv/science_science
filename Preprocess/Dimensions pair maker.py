#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 11:06:18 2021

@author: sahand
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

# dir_root = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/'
dir_root = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/'

data = pd.read_csv(dir_root+'Corpus/Dimensions - AI/corpus references',names=['refs'])
data_pub_ids = pd.read_csv(dir_root+'Corpus/Dimensions - AI/publication idx',names=['pub_id'])
data['pub_id'] = data_pub_ids['pub_id']
data = data[pd.notnull(data['refs'])]

# =============================================================================
# Make citation pairs
# =============================================================================
pairs = []
for idx,paper in tqdm(data.iterrows(),total=data.shape[0]):
    refs = paper['refs'][1:-1].replace("'",'').split(', ')
    for ref in refs:
        pairs.append([paper['pub_id'],ref])
        
pairs = pd.DataFrame(pairs,columns=['citing','cited'])
pairs.to_csv(dir_root+'Corpus/Dimensions/citations pairs',index=False)

# =============================================================================
# Filter pairs
# =============================================================================
pairs = pd.read_csv(dir_root+'Corpus/Dimensions/citations pairs')
mask = data_pub_ids['pub_id'].values.tolist()
pairs_masked = pairs[pairs['cited'].isin(mask)]
pairs_masked.to_csv(dir_root+'Corpus/Dimensions/citations pairs - masked',index=False)

# =============================================================================
# Make co-citation pairs
# =============================================================================
# refs = data['refs'].apply(lambda x: x[1:-1].replace("'",'').split(', ')).values.tolist()
# refs = [item for sublist in refs for item in sublist]
# refs.extend(data['pub_id'].values.tolist())
# refs = list( dict.fromkeys(refs) ) # remove duplicates

pairs = []
for idx,paper in tqdm(data.iterrows(),total=data.shape[0]):
    refs = paper['refs'][1:-1].replace("'",'').split(', ')
    for ref in refs:
        pairs.append([paper['pub_id'],ref])
        
pairs = pd.DataFrame(pairs,columns=['citing','cited'])
pair_groups = pairs.groupby('cited').groups
pair_groups_cited = [list(x) for x in list(pair_groups.values()) if len(list(x))>1]

# group = pair_groups_cited[0]
# for i in group:
    