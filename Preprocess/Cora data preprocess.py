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
# read JSON dblp data
# =============================================================================
# input_file = '/mnt/6016589416586D52/Users/z5204044/Documents/Dataset/Aminer/dblp_papers_v11.txt'
input_file = '/mnt/16A4A9BCA4A99EAD/DBLP/dblp.v12.json'
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
