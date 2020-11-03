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
path = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/'
labels = pd.read_csv(path+'Corpus/AI 4k/embeddings/clustering/k10/Doc2Vec patent_wos_ai corpus DEC 200,500,10 k10 labels')
years = pd.read_csv(path+'Corpus/AI 4k/embeddings/clustering/k10/years 1990-2019.csv')
df = labels.copy()
df['year'] = years['Year']
year_list = list(df.groupby('year').groups.keys())
cluster_list = list(df.groupby('label').groups.keys())

X = year_list

df['year']
Y = # list of lists


# Create data
X = np.arange(0, 10, 1)
Y = X + 5 * np.random.random((5, X.size))
 
# There are 4 types of baseline we can use:
baseline = ["zero", "sym", "wiggle", "weighted_wiggle"]
 
# Let's make 4 plots, 1 for each baseline
for n, v in enumerate(baseline):
   if n<3 :
      plt.tick_params(labelbottom='off')
   plt.subplot(2 ,2, n + 1)
   plt.stackplot(X, *Y, baseline=v)
   plt.title(v)
   plt.axis('tight', size=0.2)
