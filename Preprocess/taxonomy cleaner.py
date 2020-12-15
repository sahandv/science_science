#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:22:11 2020

@author: sahand
"""
import pandas as pd
from tqdm import tqdm

taxonomy = pd.read_csv('/home/sahand/GoogleDrive/Data/Corpus/Taxonomy/1980-2019 300k n-gram author keyword taxonomy.csv')

# =============================================================================
# Add len to columns
# =============================================================================
lens = []
for i,row in tqdm(taxonomy.iterrows(),total=taxonomy.shape[0]):
    lens.append(len(row['keywords'])-row['grams']+1)
taxonomy['charlen'] = lens
taxonomy.to_csv('/home/sahand/GoogleDrive/Data/Corpus/Taxonomy/1980-2019 300k n-gram author keyword taxonomy.csv',index=False)

# =============================================================================
# Remove unwanted rows
# =============================================================================
quantile = taxonomy['count'].quantile(0.95) # 95th percentile
taxonomy_new = taxonomy[taxonomy['count']>=quantile]
taxonomy_new = taxonomy_new[taxonomy_new['charlen']>taxonomy_new['grams']*2]
taxonomy_new['keywords'] = taxonomy_new['keywords'].apply(lambda x: ' '+x+' ')

taxonomy_new.to_csv('/home/sahand/GoogleDrive/Data/Corpus/Taxonomy/1980-2019 300k n-gram author keyword taxonomy - 95percentile and cleaned.csv',index=False)


