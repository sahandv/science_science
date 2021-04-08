#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 09:30:07 2021

@author: sahand
"""
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm

# dir_root = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/cora-classify/cora/'
dir_root = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/Corpus/cora-classify/cora/'

pairs = pd.read_csv(dir_root+'citations',names=['citing','cited'],sep='\t')
pairs.info(memory_usage='deep')

# =============================================================================
# Make co-citation pairs
# =============================================================================
pair_groups = pairs.groupby(['cited']).groups
pair_groups_list = [list(x) for x in list(pair_groups.values()) if len(list(x))>1]

pairs[pd.isna(pairs['citing'])]

# =============================================================================
# # eat memory (fast!)
# =============================================================================
pairs_cocitation = []
for x in tqdm(range(len(pair_groups_list))):
    group = pairs.iloc[pair_groups_list[x]]['citing'].values.tolist()
    n = len(group)
    for i in range(n):
        for j in range(i+1,n):
            pairs_cocitation.append((group[i],group[j])) # the int way
            # pairs_cocitation.append(group[i]+'-'+group[j]) # the string way


# del pairs
gc.collect()

pairs_cocitation = pd.DataFrame(pairs_cocitation,columns=['cite1','cite2'])
# pairs_cocitation['weight'] = 0
# pairs_cocitation.columns = ['pair','weight']
pairs_cocitation.to_csv(dir_root+'co-citation-pairs-concat.csv',index=False,header=False)

# read from file or continue to use the same one
pairs_cocitation = pd.read_csv(dir_root+'co-citation-pairs-concat.csv',names=['pair']) # str way
pairs_cocitation = pd.read_csv(dir_root+'co-citation-pairs-concat.csv',names=['cite1','cite2']) # int way
# =============================================================================
# str way
unique_co_citations = pairs_cocitation['pair'].unique()
pd.DataFrame(unique_co_citations).to_csv(dir_root+'co-citation-pairs-concat-unique.csv',index=False,header=False)

unique_co_citations_weights = pairs_cocitation['pair'].value_counts()
pd.DataFrame(unique_co_citations_weights).to_csv(dir_root+'co-citation-pairs-concat-unique-weights.csv',index=False,header=False)

# Check and combine
unique_co_citations = pd.read_csv(dir_root+'o-citation-pairs-concat-unique.csv',names=['pair'])
weights = pd.read_csv(dir_root+'co-citation-pairs-concat-unique-weights.csv',names=['weight'])['weight']
unique_co_citations['weight'] = weights
# =============================================================================
# int way
pairs_cocitation = pairs_cocitation.groupby(['cite1','cite2']).size().reset_index().rename(columns={0:'count'})
pairs_cocitation.to_csv(dir_root+'co-citation-pairs-concat-unique-weights.csv',index=False)
pairs_cocitation.sample(5)
# =============================================================================
# # save memory (slow!) - pandas method - weighted
# =============================================================================
pairs_cocitation = pd.DataFrame([],columns=['pair','weight'])
for x in tqdm(range(len(pair_groups_list))):
    group = pairs.iloc[pair_groups_list[x]]['citing'].values.tolist()
    n = len(group)
    for i in range(n):
        for j in range(i+1,n):
            if group[i]+';'+group[j] not in pairs_cocitation['pair'].values.tolist():
            # if row.any() == False:
                pairs_cocitation = pairs_cocitation.append({'pair':group[i]+';'+group[j],'weight':0},ignore_index=True)
            else:
                row = pairs_cocitation['pair']==group[i]+';'+group[j]
                pairs_cocitation.loc[row,'weight'] = pairs_cocitation[row]['weight'].values.tolist()[0]+1
                
# =============================================================================
# # save memory (slow!) - python list method -weighted
# =============================================================================
pairs_cocitation_index = []
pairs_cocitation_weight = []
for x in tqdm(range(len(pair_groups_list))):
    group = pairs.iloc[pair_groups_list[x]]['citing'].values.tolist()
    n = len(group)
    for i in range(n):
        for j in range(i+1,n):
            if (group[i]+'-'+group[j] not in pairs_cocitation_index) and (group[j]+'-'+group[i] not in pairs_cocitation_index):
                pairs_cocitation_index.append(group[i]+'-'+group[j])
                pairs_cocitation_weight.append(0)
            else:
                try:
                    pairs_cocitation_weight[pairs_cocitation_index.index(group[i]+'-'+group[j])] = pairs_cocitation_weight[pairs_cocitation_index.index(group[i]+'-'+group[j])]+1
                except:
                    try:
                        pairs_cocitation_weight[pairs_cocitation_index.index(group[j]+'-'+group[i])] = pairs_cocitation_weight[pairs_cocitation_index.index(group[j]+'-'+group[i])]+1
                    except:
                        print('WTF?! Where is ',group[j]+'-'+group[i],'or',group[i]+'-'+group[j],'then?')

pairs_cocitation_index = pd.DataFrame(pairs_cocitation_index)
pairs_cocitation_index.columns = ['pair']
pairs_cocitation_index['weight'] =  pairs_cocitation_weight
pairs_cocitation_index.to_csv(dir_root+'Corpus/Dimensions/co-citations pairs - weighted',index=False)

# =============================================================================
# # save memory (slow!) - python list method -unweighted
# =============================================================================
pairs_cocitation_index = []
pairs_cocitation_weight = []
for x in tqdm(range(len(pair_groups_list))):
    group = pairs.iloc[pair_groups_list[x]]['citing'].values.tolist()
    n = len(group)
    for i in range(n):
        for j in range(i+1,n):
            if (group[i]+'-'+group[j] not in pairs_cocitation_index) and (group[j]+'-'+group[i] not in pairs_cocitation_index):
                pairs_cocitation_index.append(group[i]+'-'+group[j])