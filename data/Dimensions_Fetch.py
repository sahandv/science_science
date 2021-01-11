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
# dimcli.login(key="my-secret-key",endpoint="https://app.dimensions.ai")
# dimcli --init
dimcli.login()
dsl = dimcli.Dsl()

# =============================================================================
# Query
# =============================================================================

# data = dsl.query("""search publications for "black holes" return publications""")
# data = dsl.query("""search publications for "black holes" return publications""", verbose=False)
years = list(range(1960,2021))
for year in tqdm(years):
    # year = 2016
    data = dsl.query_iterative(r"""search publications for "\"artificial intelligence\"" where year="""+str(year)+
                               """ and (type="article" or type="proceeding") and times_cited > 0 
                               return publications 
                               [id + authors + researchers + linkout + dimensions_url + doi + title + abstract + 
                                times_cited + altmetric + reference_ids +  year + category_for + journal + 
                               proceedings_title + publisher + research_orgs]""")

    # =============================================================================
    # Process & Save
    # =============================================================================
    
    data.json.keys()
    len(data.publications)
    print("We got", data.count_batch, "results out of", data.count_total)
     # ps errors are printed out by default
    print(data.errors_string)
    
    csv = pd.DataFrame(data.publications)
    csv.to_csv('/mnt/16A4A9BCA4A99EAD/Dimensions/'+str(year)+' dimensions AI articles-proceedings cited>0.csv')
    
    del csv
    del data
    gc.collect()
    
# =============================================================================
# Data combine
# =============================================================================
data = pd.read_csv('/mnt/6016589416586D52/Users/z5204044/Documents/Dataset/Dimensions/'+str(1960)+' dimensions AI articles-proceedings cited>0.csv',index_col=0)
years = list(range(1961,2021))
for year in tqdm(years):
    tmp = pd.read_csv('/mnt/6016589416586D52/Users/z5204044/Documents/Dataset/Dimensions/'+str(year)+' dimensions AI articles-proceedings cited>0.csv',index_col=0)
    data = data.append(tmp,ignore_index=True)
data.to_csv('/mnt/6016589416586D52/Users/z5204044/Documents/Dataset/Dimensions/all dimensions AI articles-proceedings cited>0.csv')