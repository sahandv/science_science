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

data = pd.read_csv(dir_root+'Corpus/Dimensions/corpus references',names=['refs'])
data_pub_ids = pd.read_csv(dir_root+'Corpus/Dimensions/publication idx',names=['pub_id'])
data['pub_id'] = data_pub_ids['pub_id']
data = data[pd.notnull(data['refs'])]

# =============================================================================
# Make pairs
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