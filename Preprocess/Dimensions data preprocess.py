#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:30:30 2021

@author: sahand
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
dir_root = '/home/sahand/GoogleDrive/Data/Corpus/Dimensions/' # ryzen
# dir_root = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/Dimensions/' # c1314

# =============================================================================
# Label prep - cleanup of data without label
# =============================================================================
categories = pd.read_csv(dir_root+'corpus category_for')
data = pd.read_csv(dir_root+'publication idx')
data['cat'] = categories['category_for']
# data['cat'] = data.cat.str.replace('[','').str.replace(']','').str[1:-1].str.split('}, {')
# data['cat1'] = data['cat'][0]
pub_ids = pd.DataFrame(data['id'])
data = data[pd.notnull(data['cat'])]
categories = categories[pd.notnull(categories['category_for'])]
categories.to_csv(dir_root+'corpus category_for',index=False,header=False)
pub_ids_mask = pd.DataFrame(data['id'])
pub_ids_mask.to_csv(dir_root+'publication idx',index=False,header=False)
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
name_template = ['cat_id','cat_name','for_id','for_name','for_id_root']
column_names =  []
for i in range(int(len(cats_all_df.columns)/len(name_template))):
    column_names.extend([x+'-'+str(i) for x in name_template])

cats_all_df.columns = column_names
# cats_all_df.to_csv(dir_root+'categories_procesed_further')

cat_0_groups = cats_all_df.groupby('for_id_root-0').size().reset_index()#.agg(['count'])
print(cat_0_groups)
cat_1_groups = cats_all_df.groupby('for_id_root-1').size().reset_index()#.agg(['count'])
print(cat_1_groups)
cat_2_groups = cats_all_df.groupby('for_id-2').size().reset_index()#.agg(['count'])
print(cat_2_groups) 


def category_selector(cat_row,level:int=0,max_level:int=7):
    """
    It is an recursive function to select the categories.
    
    Parameters
    ----------
    cat_row : Pandas Series
        A row of categories.
    level : int
        Depth of recurring function
    max_level : int
        max depth of recurring  function
    
        
    Returns
    -------
    cat :  list
        Category
    
    """
    cat = row['for_id_root-'+str(level)]

    if (cat!='01' and cat!='06' and cat!='09' and cat!='10' and
        cat!='11' and cat!='15' and cat!='17') is False:
        return cat
    else:
        if level<max_level:
            return category_selector(row,level+1,max_level)
            

cats = []
for i,row in tqdm(cats_all_df.iterrows(),total=cats_all_df.shape[0]):
    cats.append(category_selector(row,0,7))

cats = pd.DataFrame(cats,columns=['category'])
if cats[pd.isnull(cats['category'])].shape[0]>0:
    print('oops! found some unwanted categories...')
cats.to_csv(dir_root+'categories_masked_clean',index=False)

# eight = cats_all_df[((cats_all_df['for_id_root-0']!='01') & 
#                      (cats_all_df['for_id_root-0']!='06') &
#                      (cats_all_df['for_id_root-0']!='09') &
#                      (cats_all_df['for_id_root-0']!='10') &
#                      (cats_all_df['for_id_root-0']!='11') &
#                      (cats_all_df['for_id_root-0']!='15') &
#                      (cats_all_df['for_id_root-0']!='17'))]



# =============================================================================
# Check labels and etc.
# =============================================================================
cats = pd.read_csv(dir_root+'categories_masked_clean_categorical', dtype=str)
cats.info()
cats.category = pd.Categorical(cats.category)
cats.category = cats.category.cat.codes
cats.to_csv(dir_root+'categories_masked_clean_categorical',index=False)



