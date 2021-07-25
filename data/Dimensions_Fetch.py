#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 11:10:11 2020

This script is for fetching dimensions data

@author: sahand
"""
import gc
from tqdm import tqdm
import dimcli
import pandas as pd
import numpy as np
dimcli.login(key="CC1FEFF8637149DBBA873831FCA4471F",endpoint="https://app.dimensions.ai")
# dimcli --init
# dimcli.login()
dsl = dimcli.Dsl()

# =============================================================================
# Query - get year by year
# =============================================================================

# data = dsl.query("""search publications for "black holes" return publications""")
# data = dsl.query("""search publications for "black holes" return publications""", verbose=False)
years = list(range(1960,2021))
for year in tqdm(years):
    # year = 2016
    # data = dsl.query_iterative(r"""search publications for "\"artificial intelligence\"" where year="""+str(year)+
    #                            """ and (type="article" or type="proceeding") and times_cited > 0 
    #                            return publications 
    #                            [id + authors + researchers + linkout + dimensions_url + doi + title + abstract + 
    #                             times_cited + altmetric + reference_ids +  year + category_for + journal + 
    #                            proceedings_title + publisher + research_orgs]""")
    data = dsl.query_iterative(r"""search publications in full_data for "(\"machine learning\" OR \"artificial intelligence\" OR \"deep learning\" OR \"neural networks\" OR \"pattern recognition\")" where year="""+str(year)+
                               """ and (type="article" or type="proceeding") and 
                               (category_for.name="01 Mathematical Sciences" or category_for.name="09 Engineering" or category_for.name="11 Medical and Health Sciences" or 
                                category_for.name="17 Psychology and Cognitive Sciences" or category_for.name="06 Biological Sciences" or 
                                category_for.name="15 Commerce, Management, Tourism and Services" or category_for.name="10 Technology")
                               and abstract is not empty and reference_ids is not empty and title is not empty and category_for is not empty
                               return publications 
                               [id + authors + researchers + linkout + dimensions_url + doi + title + abstract + concepts + concepts_scores + 
                                times_cited + altmetric + reference_ids +  year + category_for + journal + 
                               proceedings_title + publisher + research_orgs] sort by altmetric desc""") #for "(\"machine learning\" OR \"artificial intelligence\" OR \"deep learning\")"

    # =============================================================================
    # Process & Save
    # =============================================================================
    
    data.json.keys()
    len(data.publications)
    print("We got", data.count_batch, "results out of", data.count_total)
     # ps errors are printed out by default
    print(data.errors_string)
    
    csv = pd.DataFrame(data.publications)
    csv.to_csv('/home/sahand/Dimensions/'+str(year)+' dimensions AI articles-proceedings cited>0.csv')
    
    del csv
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
                               [id + authors + researchers + linkout + dimensions_url + doi + title + abstract + 
                                times_cited + altmetric + reference_ids +  year + category_for + journal + 
                               proceedings_title + publisher + research_orgs] sort by altmetric desc""") # for "\"machine learning\""  and times_cited > 0 
                               
data.json.keys()
len(data.publications)
print("We got", data.count_batch, "results out of", data.count_total)
 # ps errors are printed out by default
print(data.errors_string)

csv = pd.DataFrame(data.publications)
csv.to_csv('/mnt/16A4A9BCA4A99EAD/Dimensions/dimensions AI DL ML all articles-proceedings.csv')

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
# Data combine
# =============================================================================
data = pd.read_csv('/mnt/16A4A9BCA4A99EAD/Dimensions/'+str(1960)+' dimensions AI articles-proceedings cited>0.csv',index_col=0)
years = list(range(1961,2021))
for year in tqdm(years):
    tmp = pd.read_csv('/mnt/16A4A9BCA4A99EAD/Dimensions/'+str(year)+' dimensions AI articles-proceedings cited>0.csv',index_col=0)
    data = data.append(tmp,ignore_index=True)
data.to_csv('/mnt/16A4A9BCA4A99EAD/Dimensions/all dimensions AI ML DL articles-proceedings cited>4.csv')