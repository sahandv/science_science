#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:32:39 2019

@author: github.com/sahandv
"""

import sys, os, time
import platform
from pathlib import Path


def find_first_big_csv(filename,chunksize,names,column,needle):
    import pandas as pd
    for chunk in pd.read_csv(filename, chunksize=chunksize,names=names):
        result = chunk[chunk[column].str.contains(needle)]
        if result.shape[0]>0:
            return result.index[0]
    return False
        

def unify_citing(input_file,destination_file,chunksize,column,names,names_new,verbose = False):
# =============================================================================
#     neme_new : name of value column for groups (it is the index of first occurrence)
#     column : name of column to group
#     names : names of columns of the source file
# =============================================================================
    import pandas as pd
    first = True
    with open(destination_file, 'a') as f:
        for chunk in pd.read_csv(input_file, chunksize=chunksize,names=names):
            df = pd.DataFrame(chunk.groupby(column).apply(lambda x: x.index.tolist()[0]))#.first()
            df.columns = names_new
            if verbose == True:
                print(df.iloc[-1][names_new[0]])
            if first is True:
                df.to_csv(f)
                first = False
            else:
                df.to_csv(f, header=False)