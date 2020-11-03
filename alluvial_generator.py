#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 17:59:55 2020

@author: sahand
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Data prep
# =============================================================================
path = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/'
labels = pd.read_csv(path+'Corpus/AI 4k/embeddings/clustering/k10/Doc2Vec patent_wos_ai corpus DEC 200,500,10 k10 labels')
years = pd.read_csv(path+'Corpus/AI 4k/embeddings/clustering/k10/years 1990-2019.csv')
df = labels.copy()
df['year'] = years['Year']
year_list = list(df.groupby('year').groups.keys())
cluster_list = list(df.groupby('label').groups.keys())

X = year_list

Y = pd.DataFrame({'label':[],'count':[]})
for year in year_list:
    Y1 = pd.DataFrame(df[df['year']==year].groupby('label').agg(['count'])['year']['count']).reset_index()
    Y = (pd.merge(Y, Y1,on='label', how='outer').fillna(0))

Y.columns = ['zero','label']+list(range(len(list(Y.columns))-2))
Y = Y.drop(['zero','label'],axis=1)
Y = Y.values


labels = ["c 1 ", "c 2", "c 3","c 4 ", "c 5", "c 6","c 7 ", "c 8", "c 9", "c 10"]
fig, ax = plt.subplots(figsize=(15,7))
ax.stackplot(X, *Y, baseline="weighted_wiggle",labels=labels)
ax.legend(loc='upper left')
# plt.figure()
plt.show()
