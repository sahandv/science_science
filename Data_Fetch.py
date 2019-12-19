#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:16:21 2019

@author: github.com/sahandv
"""

from sciosci import scopus_helper as sh
from sciosci.assets import generic_assets as ga

# =============================================================================
# Initialization of the package
# =============================================================================
sh.scopus_initialize()
ga.general_initialize()

from sciosci.assets import keyword_assets as kw
from sciosci.assets import advanced_assets as aa
import pandas as pd
import numpy as np
from tqdm import tqdm

# =============================================================================
# https://www.springer.com/gp/product-search/discipline?disciplineId=ai&dnc=true&facet-lan=lan__en&facet-subj=subj__I21000&facet-type=type__journal&page=4&returnUrl=gp%2Fcomputer-science%2Fai&topic=I21000%2CI21010%2CI21020%2CI21030%2CI21040%2CI21050%2CI21060
# =============================================================================
source_list = pd.read_csv("data/source_list.csv",delimiter='__',names=['source'])
for source in tqdm(source_list['source'].values.tolist()[21:30]):
    print('Getting data for: ',source)
    try:
        sh.search_scopus('SRCTITLE("'+source+'")',download=True)
    except:
        raise Exception('Error while getting data from Scopus')
