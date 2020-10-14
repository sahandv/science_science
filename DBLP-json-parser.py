#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:25:40 2020

@author: sahand
"""

from sciosci.assets import text_assets as ta
import json   
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

# input_file = '/mnt/6016589416586D52/Users/z5204044/Documents/Dataset//Aminer/dblp_papers_v11.txt'
input_file = '/mnt/6016589416586D52/Users/z5204044/Documents/Dataset//Aminer/dblp.v12.json'
# input_file = '/mnt/6016589416586D52/Users/z5204044/Documents/Dataset/Aminer/dblp.v10/dblp-ref/dblp-ref-0.json'

# from sciosci import scopus_helper as sh
# df = sh.json_to_df(input_file,skip=[0])

papers = []
count = 0
stop = 10000
with open(input_file) as f:
    for line in f:
        count+=1
        if count > stop:
            break
        try:
            papers.append(json.loads(line))
        except:
            try:
                line = line[1:]
                papers.append(json.loads(line))
            except:
                print('no way to parse the line ',count-1)
                

papers = pd.DataFrame(papers)
papers['abstract'] = papers['indexed_abstract'].progress_apply(
    lambda x: ta.indexed_text_to_string(x['InvertedIndex'],x['IndexLength']) if pd.notnull(x) else np.nan)

