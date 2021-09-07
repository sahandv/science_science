#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:22:11 2020

@author: sahand
"""
import pandas as pd
from tqdm import tqdm
from sciosci.assets import text_assets as ta
from sciosci.assets import keyword_dictionaries as kd
tqdm.pandas()

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

# =============================================================================
# CS Ontology 
# =============================================================================
cso = pd.read_csv('/home/sahand/GoogleDrive/Data/Corpus/Taxonomy/CSO.3.3.csv',names=['a','relation','b'])

cso.loc[cso['relation'].str.contains("relatedEquivalent"),'relation'] = 'equal'
cso.loc[cso['relation'].str.contains("superTopicOf"),'relation'] = 'parent_of'
cso.loc[cso['relation'].str.contains("contributesTo"),'relation'] = 'are-related'
cso.loc[cso['relation'].str.contains("type"),'relation'] = 'type'
cso.loc[cso['relation'].str.contains("label"),'relation'] = 'readable-label'
cso.loc[cso['relation'].str.contains("preferentialEquivalent"),'relation'] = 'preferred-name-is'
cso.loc[cso['relation'].str.contains("sameAs"),'relation'] = 'external-source-entity'
cso.loc[cso['relation'].str.contains("relatedLink"),'relation'] = 'link'

#1 replace by human readable labels
label_dict = cso[cso['relation']=='readable-label'][['a','b']].reset_index(drop=True).values

tmp = {}
for row in tqdm(label_dict):
    tmp[row[0]]=row[1]
label_dict = tmp
del tmp

tmp = []
for i,row in tqdm(cso.iterrows(),total=cso.shape[0]):
    a = row['a']
    b = row['b']
    for bad, good in label_dict.items():
        a = a.replace(bad,good)
        b = b.replace(bad,good)
    tmp.append([a,row['relation'],b])
    
cso_with_names = pd.DataFrame(tmp,columns=cso.columns)
cso_with_names.to_csv('/home/sahand/GoogleDrive/Data/Corpus/Taxonomy/CSO.3.3-with-labels.csv',index=False)

#2 extract taxonomy
cso_taxonomy = pd.DataFrame(list(set(label_dict.values())),columns=['keywords'])
cso_taxonomy['keywords'] = cso_taxonomy['keywords'].str.replace('@en .','')
cso_taxonomy['keywords'] = cso_taxonomy['keywords'].progress_apply(lambda x: ta.replace_british_american(x,kd.gb2us)) # Optional step
cso_taxonomy.to_csv('/home/sahand/GoogleDrive/Data/Corpus/Taxonomy/CSO.3.3-taxonomy-US.csv',index=False)


