#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 11:10:11 2020

This script is for fetching dimensions data

@author: sahand
"""

import dimcli
import pandas as pd
# dimcli.login(key="my-secret-key",endpoint="https://app.dimensions.ai")
# dimcli --init
dimcli.login()
dsl = dimcli.Dsl()

# =============================================================================
# Query
# =============================================================================

data = dsl.query("""search publications for "black holes" return publications""")
data = dsl.query("""search publications for "black holes" return publications""", verbose=False)
data = dsl.query_iterative("""search publications for "artificial intelligence" where year=2016 and times_cited > 30 return publications 
                           [id + authors + linkout + doi + title + abstract + reference_ids + year + category_bra + category_for]""")

# =============================================================================
# Process & Save
# =============================================================================

data.json.keys()
len(data.publications)
print("We got", data.count_batch, "results out of", data.count_total)
 # ps errors are printed out by default
print(data.errors_string)

csv = pd.DataFrame(data.publications)
csv.to_csv('~/dimensions AI 1990 cited>10.csv')
