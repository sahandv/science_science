#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:12:51 2020

@author: sahand

"""
# =============================================================================
# read JSON dblp data
# =============================================================================
dir_path = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/Corpus/cora-classify/cora/'

import json   
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
tqdm.pandas()
from sciosci.assets import text_assets

papers_list_raw = pd.read_csv(dir_path+'papers',sep='\t',names=['id','filename','citation string']) # contains duplicates
# papers_list_raw = papers_list_raw.groupby('id').first().reset_index()

papers_list = pd.read_csv(dir_path+'classifications',sep='\t',names=['paper_name','class'])

citations = pd.read_csv(dir_path+'citations',names=['referring_id','cited_id'],sep='\t')

