#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:12:51 2020

@author: sahand


References:
Jie Tang, Jing Zhang, Limin Yao, Juanzi Li, Li Zhang, and Zhong Su. ArnetMiner: Extraction and Mining of Academic Social Networks. In Proceedings of the Fourteenth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (SIGKDD'2008). pp.990-998.
Arnab Sinha, Zhihong Shen, Yang Song, Hao Ma, Darrin Eide, Bo-June (Paul) Hsu, and Kuansan Wang. 2015. An Overview of Microsoft Academic Service (MAS) and Applications. In Proceedings of the 24th International Conference on World Wide Web (WWW ’15 Companion). ACM, New York, NY, USA, 243-246. [PDF][System][API]  ***For V10, V11, V12***

"""
# =============================================================================
# read  dblp data
# =============================================================================
# input_file = '/mnt/6016589416586D52/Users/z5204044/Documents/Dataset/Aminer/dblp_papers_v11.txt'
input_file = '/mnt/6016589416586D52/Users/z5204044/Documents/Dataset/Aminer/dblp.v12.json'
# input_file = '/mnt/6016589416586D52/Users/z5204044/Documents/Dataset/Aminer/dblp.v10/dblp-ref/dblp-ref-0.json'

# from sciosci import scopus_helper as sh
# df = sh.json_to_df(input_file,skip=[0])

import json   
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
tqdm.pandas()
from sciosci.assets import text_assets

empty_venue = {'raw': ''}
count = 0
pace = 400000 # larger pace will be faster, but mind the memory. 400k is safe for 16GB
done = False
papers_df = pd.DataFrame([])
total = 0
print('\nInspecting the file...')
with open(input_file) as f:
    for line in f:
        total+=1
print('\nDone inspecting',total,'lines')

while not done:
    start = count
    count = 0
    stop = min(start+pace,total)
    papers = []
    gc.collect()
    print('\n\nWorking from',start,'to',stop)
    with open(input_file) as f:
        for line in tqdm(f,total=stop):
            count+=1
            if count < start:
                continue
            if count > stop:
                break
            try:
                papers.append(json.loads(line))
                # year = json.loads(line)['year']
                # years.append(year)
            except:
                try:
                    line = line[1:]
                    papers.append(json.loads(line))
                    # year = json.loads(line)['year']
                    # years.append(year)
                except:
                    print('\n*No way to parse the line ',count-1,'!')
            if count>=total:
                done = True
                break
            
    # papers_df = pd.concat([pd.DataFrame(papers), papers_df])
    papers_df = pd.DataFrame(papers)
    print('\nPreparing the dataframe for storage...')
    papers_df['abstract'] = papers_df['indexed_abstract'].progress_apply(
        lambda x: text_assets.indexed_text_to_string(x['InvertedIndex'],x['IndexLength']) if pd.notnull(x) else np.nan)
    # papers_df['fos'] = papers_df['fos'].progress_apply(
    #     lambda x: pd.DataFrame(x).values.tolist() if np.all(pd.notnull(x)) else np.nan)
    papers_df['fos'] = papers_df['fos'].progress_apply(
        lambda x: json.dumps(x) if np.all(pd.notnull(x)) else np.nan)
    papers_df['authors'] = papers_df['authors'].progress_apply(
        lambda x: json.dumps(x) if np.all(pd.notnull(x)) else np.nan)
    
    
    papers_df['venue'] = papers_df['venue'].progress_apply(
        lambda x: x if np.all(pd.notnull(x)) else empty_venue)
    venue = pd.DataFrame(papers_df['venue'].values.tolist())
    
    venue.columns=['venue_name','venue_id','venue_type']
    papers_df = pd.concat([papers_df, venue], axis=1)
    
    papers_df = papers_df.drop('venue',axis=1)
    papers_df = papers_df.drop('indexed_abstract',axis=1)
    print('\nPreparation done. Saving to disk...')
    if start==0:
        papers_df.to_csv('/mnt/6016589416586D52/Users/z5204044/Documents/'
                                +'Dataset/Aminer/dblp.v12/'+str(start)+'-'+str(stop)+'.csv',index=False)
    else:
        papers_df.to_csv('/mnt/6016589416586D52/Users/z5204044/Documents/'
                                +'Dataset/Aminer/dblp.v12/'+str(start)+'-'+str(stop)+'.csv',index=False,header=False)
    print('\nSaved to disk.')