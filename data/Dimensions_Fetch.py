#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 11:10:11 2020

This script is for fetching dimensions data

@author: sahand
"""

import dimcli
# dimcli.login(key="my-secret-key",endpoint="https://app.dimensions.ai")
# dimcli --init
dimcli.login()
dsl = dimcli.Dsl()

data = dsl.query("""search publications for "black holes" return publications""")
data = dsl.query("""search publications for "black holes" return publications""", verbose=False)
data = dsl.query_iterative("""search publications for "black holes" where year=1990 and times_cited > 10 return publications""")


data.json.keys()
len(data.publications)
print("We got", data.count_batch, "results out of", data.count_total)
 # ps errors are printed out by default
data = dsl.query("""search publications for "black holes" return spaceships""")
print(data.errors_string)


