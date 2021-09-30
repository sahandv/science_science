#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 11:10:11 2020

This script is for fetching dimensions data

@author: sahand
"""
import os
import gc
from tqdm import tqdm
import dimcli
import pandas as pd
import numpy as np
import json
dimcli.login(key="CC1FEFF8637149DBBA873831FCA4471F",endpoint="https://app.dimensions.ai")
# dimcli --init
# dimcli.login()
dsl = dimcli.Dsl()

# =============================================================================
# Query - get year by year
# =============================================================================

# data = dsl.query("""search publications for "black holes" return publications""")
# data = dsl.query("""search publications for "black holes" return publications""", verbose=False)
years = list(range(2018,2021))
for year in tqdm(years):
    # year = 2016
    # data = dsl.query_iterative(r"""search publications for "\"artificial intelligence\"" where year="""+str(year)+
    #                            """ and (type="article" or type="proceeding") and times_cited > 0 
    #                            return publications 
    #                            [id + authors + researchers + linkout + dimensions_url + doi + title + abstract + 
    #                             times_cited + altmetric + reference_ids +  year + category_for + journal + 
    #                            proceedings_title + publisher + research_orgs]""")
    data = dsl.query_iterative(r"""search publications in full_data for "(\"machine learning\" OR \"artificial intelligence\" OR \"deep learning\")"  where year="""+str(year)+
                               """ and (type="article" or type="proceeding") and times_cited <1 and
                               (category_for.name="01 Mathematical Sciences" or category_for.name="09 Engineering" or category_for.name="11 Medical and Health Sciences" or 
                                category_for.name="17 Psychology and Cognitive Sciences" or category_for.name="06 Biological Sciences" or 
                                category_for.name="15 Commerce, Management, Tourism and Services" or category_for.name="10 Technology")
                               and abstract is not empty and reference_ids is not empty and title is not empty and category_for is not empty
                               return publications 
                               [id + authors + dimensions_url + doi + title + abstract + concepts_scores + authors_count + terms +
                                times_cited + reference_ids +  year + category_for +  category_ua + journal] """) #for "(\"machine learning\" OR \"artificial intelligence\" OR \"deep learning\")"

# [id + authors + researchers + linkout + dimensions_url + doi + title + abstract + concepts_scores + pages + funders + authors_count + mesh_terms + terms +
#  times_cited + altmetric + reference_ids +  year + category_for + category_bra + category_hra + category_rcdc + journal +  category_ua + FOR_first + FOR +
# proceedings_title + publisher + research_orgs ]
    # =============================================================================
    # Process & Save
    # =============================================================================
    
    output_address = '/home/sahand/GoogleDrive/Data/Corpus/Dimension All/'+str(year)+' dimensions AI articles-proceedings cited<1.json'
    
    # data_json = json.dumps(data.publications)

    with open(output_address, 'w') as json_file:
        json.dump(data.publications, json_file)
    
    # data.json.keys()
    # len(data.publications)
    # print("We got", data.count_batch, "results out of", data.count_total)
    #  # ps errors are printed out by default
    # print(data.errors_string)
    
    # csv = pd.DataFrame(data.publications)
    # csv.to_csv('/home/sahand/GoogleDrive/Data/Corpus/Dimension All/'+str(year)+' dimensions AI articles-proceedings cited>0.csv')
    
    # del csv
    del data
    gc.collect()
    
# =============================================================================
# Query - get by category
# =============================================================================
categories = ["01 Mathematical Sciences","09 Engineering","11 Medical and Health Sciences",
              "17 Psychology and Cognitive Sciences","06 Biological Sciences",
              "15 Commerce, Management, Tourism and Services","10 Technology"]#,
              # "13 Education","20 Language, Communication and Culture","16 Studies in Human Society"]
              

data = dsl.query_iterative(r"""search publications in full_data for "(\"machine learning\" OR \"artificial intelligence\" OR \"deep learning\")"  where year>1969
                               and (type="article" or type="proceeding") and 
                               (category_for.name="01 Mathematical Sciences" or category_for.name="09 Engineering" or category_for.name="11 Medical and Health Sciences" or 
                                category_for.name="17 Psychology and Cognitive Sciences" or category_for.name="06 Biological Sciences" or 
                                category_for.name="15 Commerce, Management, Tourism and Services" or category_for.name="10 Technology")
                               and abstract is not empty and reference_ids is not empty and title is not empty and category_for is not empty
                               return publications 
                               [id + authors + researchers + linkout + dimensions_url + doi + title + abstract 
                                times_cited + altmetric + reference_ids + references + year + category_for + journal + 
                               proceedings_title + publisher + research_orgs] sort by altmetric desc""") # for "\"machine learning\""  and times_cited > 0 
           
data.json.keys()
len(data.publications)
print("We got", data.count_batch, "results out of", data.count_total)
 # ps errors are printed out by default
print(data.errors_string)

csv = pd.DataFrame(data.publications)
csv.to_csv('/mnt/16A4A9BCA4A99EAD/Dimensions/dimensions AI DL ML all articles-proceedings.csv')

# =============================================================================
# Read for test
# =============================================================================
address = '/home/sahand/GoogleDrive/Data/Corpus/Dimension All/1961-2017 dimensions AI articles-proceedings.json'
with open(address) as f:
    data = json.load(f)
df = pd.DataFrame(data)
df_small = df[['category_for','FOR','FOR_first','year','journal']]
df_small['journal'] = df_small['journal'].apply(lambda x: {'id':np.nan,'title':np.nan} if pd.isnull(x) else x)
df_small['journal_n'] = [x['id'] for x in df_small['journal'].values.tolist()]
sample = df.sample(10000)
columns = df.columns

df_small_g = df_small.groupby('journal_n').groups
df_small_g_k = list(df_small_g.keys())
df_small_g_v = list(df_small_g.values())
# To show that the journal data category is not FOR categories in this dataset, confirming the statement in the documentations.
journal_x_data = df.loc[df_small_g_v[21]]
journal_x_data.to_csv('/home/sahand/GoogleDrive/Data/Corpus/Dimension All/category_fro_proof.csv')
# =============================================================================
# Json combine by year 
# =============================================================================
all_data = []
years = list(range(1961,2018))
for year in tqdm(years):
    with open('/home/sahand/GoogleDrive/Data/Corpus/Dimension All/'+str(year)+' dimensions AI articles-proceedings.json') as f:
        data = json.load(f)
    all_data = all_data+data
    
df = pd.DataFrame(all_data)
sample = df.sample(1000)
df['year'].hist(bins=50)

with open('/home/sahand/GoogleDrive/Data/Corpus/Dimension All/1961-2017 dimensions AI articles-proceedings.json', 'w') as json_file:
    json.dump(all_data, json_file)
# =============================================================================
# Json combine all 
# =============================================================================
list_dirs = os.listdir(path=r"/home/sahand/GoogleDrive/Data/Corpus/Dimension All/")
list_dirs = [x for x in list_dirs if x.split('.')[-1]=='json']

all_data = []
for file in tqdm(list_dirs):
    with open('/home/sahand/GoogleDrive/Data/Corpus/Dimension All/'+str(file)) as f:
        data = json.load(f)
    all_data = all_data+data
    
df = pd.DataFrame(all_data)
sample = df.sample(1000)
df['year'].hist(bins=60)

with open('/home/sahand/GoogleDrive/Data/Corpus/Dimension All/1961-2020 dimensions AI articles-proceedings.json', 'w') as json_file:
    json.dump(all_data, json_file)
# =============================================================================
# Drop randomly to reduce dataset size
# =============================================================================
np.random.seed(10)
df = pd.read_csv('/mnt/16A4A9BCA4A99EAD/Dimensions/dimensions AI DL ML all articles-proceedings cited>4.csv')
remove_n = 50000
drop_indices = np.random.choice(df.index, remove_n, replace=False)
df_subset = df.drop(drop_indices)
csv.to_csv('/mnt/16A4A9BCA4A99EAD/Dimensions/dimensions AI DL ML all articles-proceedings cited>4 reduced.csv')

# =============================================================================
# Data combine /mnt/6016589416586D52/Users/z5204044/Documents/Dataset/Dimensions_new/
# =============================================================================
data = pd.read_csv('/home/sahand/GoogleDrive/Data/Corpus/Dimension All/'+str(1960)+' dimensions AI articles-proceedings.csv',index_col=0)
years = list(range(1961,2018))
for year in tqdm(years):
    tmp = pd.read_csv('/mnt/6016589416586D52/Users/z5204044/Documents/Dataset/Dimensions_new/'+str(year)+' dimensions AI articles-proceedings.csv',index_col=0)
    data = data.append(tmp,ignore_index=True)
data.to_csv('/mnt/6016589416586D52/Users/z5204044/Documents/Dataset/Dimensions_new/all dimensions AI articles-proceedings.csv')

# =============================================================================
# Data combine /mnt/6016589416586D52/Users/z5204044/Documents/Dataset/Dimensions_new/
# =============================================================================
data = pd.read_csv('/mnt/6016589416586D52/Users/z5204044/Documents/Dataset/Dimensions_new/'+str(year)+' dimensions AI articles-proceedings cited>2.csv',index_col=0)
tmp = pd.read_csv('/mnt/6016589416586D52/Users/z5204044/Documents/Dataset/Dimensions_new/'+str(year)+' dimensions AI articles-proceedings cited=0.csv',index_col=0)
data = data.append(tmp,ignore_index=True)
data.to_csv('/mnt/6016589416586D52/Users/z5204044/Documents/Dataset/Dimensions_new/'+str(year)+' dimensions AI articles-proceedings.csv')