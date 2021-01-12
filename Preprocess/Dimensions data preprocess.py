#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:30:30 2021

@author: sahand
"""
import pandas as pd

# =============================================================================
# Label prep - cleanup of data without label
# =============================================================================
categories = pd.read_csv('/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/Dimensions/corpus category_for',names=['cat'])
data = pd.read_csv('/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/Dimensions/publication idx',names=['id'])
data['cat'] = categories['cat']
# data['cat'] = data.cat.str.replace('[','').str.replace(']','').str[1:-1].str.split('}, {')
# data['cat1'] = data['cat'][0]
pub_ids = pd.DataFrame(data['id'])
data = data[pd.notnull(data['cat'])]
categories = categories[pd.notnull(categories['cat'])]
categories.to_csv('/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/Dimensions/with label/corpus category_for',index=False,header=False)
pub_ids_mask = pd.DataFrame(data['id'])
pub_ids_mask.to_csv('/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/Dimensions/with label/publication idx',index=False,header=False)
pub_ids_mask = data['id'].values.tolist()

# filtering operation:
f_name = 'abstract_title deflemm'
corpus = pd.read_csv('/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/Dimensions/'+f_name,names=['data'])
corpus['id'] = pub_ids['id']
corpus = corpus[corpus['id'].isin(pub_ids_mask)]
corpus = corpus.drop('id',axis=1)
corpus.to_csv('/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/Dimensions/with label/'+f_name,index=False,header=False)

# =============================================================================
# Label prep - separate labels and clean
# =============================================================================
categories = pd.read_csv('/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/Dimensions/corpus category_for',names=['cat'])
data = pd.read_csv('/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/Dimensions/publication idx',names=['id'])
data['cat'] = categories['cat']
data['cat'] = data.cat.str.replace('[','').str.replace(']','').str[1:-1].str.split('}, {')

def remove_digit(string):
    return ''.join(i for i in string if not i.isdigit())
def get_digit(string):
    return ''.join(i for i in string if i.isdigit())
def clean_cats(cat_string):
    # cat_string = "'id': '3484', 'name': '1702 Cognitive Sciences'"
    cat_string = cat_string.split(", '")
    result = [x.split(': ')[1] for x in cat_string]
    result.append(get_digit(result[1]))
    result.append(remove_digit(result[1]))
    return result

    
cat_1 = [clean_cats(x[0]) for x in data['cat'].values.tolist()]
cat_1 = pd.DataFrame(cat_1)
cat_1.columns = ['cat_id','cat_name','for_id','for_name']
cat_1['cat_id'] = cat_1['cat_id'].str.replace("'","").str.strip()
cat_1['cat_name'] = cat_1['cat_name'].str.replace("'","").str.strip().str.lower()
cat_1['for_id'] = cat_1['for_id'].str.replace("'","").str.strip()
cat_1['for_name'] = cat_1['for_name'].str.replace("'","").str.strip().str.lower()
cat_1['for_id_root'] = cat_1['for_id'].str[:2]

cat_1.to_csv('/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/Dimensions/categories_processed',index=False)