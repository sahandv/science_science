#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:30:30 2021

@author: sahand
"""
import pandas as pd
from tqdm import tqdm
dir_root = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/Corpus/Dimensions/' # ryzen
# dir_root = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/Dimensions/' # c1314

# =============================================================================
# Label prep - cleanup of data without label
# =============================================================================
categories = pd.read_csv(dir_root+'corpus category_for',names=['cat'])
data = pd.read_csv(dir_root+'publication idx',names=['id'])
data['cat'] = categories['cat']
# data['cat'] = data.cat.str.replace('[','').str.replace(']','').str[1:-1].str.split('}, {')
# data['cat1'] = data['cat'][0]
pub_ids = pd.DataFrame(data['id'])
data = data[pd.notnull(data['cat'])]
categories = categories[pd.notnull(categories['cat'])]
categories.to_csv(dir_root+'with label/corpus category_for',index=False,header=False)
pub_ids_mask = pd.DataFrame(data['id'])
pub_ids_mask.to_csv(dir_root+'with label/publication idx',index=False,header=False)
pub_ids_mask = data['id'].values.tolist()

# filtering operation:
f_name = 'abstract_title deflemm'
corpus = pd.read_csv(dir_root+''+f_name,names=['data'])
corpus['id'] = pub_ids['id']
corpus = corpus[corpus['id'].isin(pub_ids_mask)]
corpus = corpus.drop('id',axis=1)
corpus.to_csv(dir_root+'with label/'+f_name,index=False,header=False)

# =============================================================================
# Label prep - separate labels and clean
# =============================================================================

categories = pd.read_csv(dir_root+'corpus category_for',names=['cat'])
data = pd.read_csv(dir_root+'publication idx',names=['id'])
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

# get the first cat only 
   
cat_1 = [clean_cats(x[0]) for x in data['cat'].values.tolist()]
cat_1 = pd.DataFrame(cat_1)
cat_1.columns = ['cat_id','cat_name','for_id','for_name']
cat_1['cat_id'] = cat_1['cat_id'].str.replace("'","").str.strip().apply(str)
cat_1['cat_name'] = cat_1['cat_name'].str.replace("'","").str.strip().str.lower()
cat_1['for_id'] = cat_1['for_id'].str.replace("'","").str.strip().apply(str)
cat_1['for_name'] = cat_1['for_name'].str.replace("'","").str.strip().str.lower()
cat_1['for_id_root'] = cat_1['for_id'].str[:2].apply(str)

cat_1.to_csv(dir_root+'categories_processed',index=False)
pd.DataFrame(cat_1['for_id_root']).to_csv(dir_root+'corpus classes_1 root',index=False,header=False)
pd.DataFrame(cat_1['for_id']).to_csv(dir_root+'corpus classes_1',index=False,header=False)

# cleanup all cats
cats_all = []
for cats in tqdm(data['cat'].values.tolist()):
    cats_tmp = []
    for cat in cats:
        cat = clean_cats(cat) # clean cat strings
        cat = [x.replace("'","").strip() for x in cat] # clean cat strings
        cat.append(cat[2][:2]) # get root
        cats_tmp.extend(cat)
    cats_all.append(cats_tmp)
    
    # cats_all.append([clean_cats(x) for x in cats])

cats_all_df = pd.DataFrame(cats_all)



# =============================================================================
# Check labels and etc.
# =============================================================================
cats = pd.read_csv(dir_root+'categories_processed', dtype=str)
cats_groups = cats.groupby('for_id_root').groups



