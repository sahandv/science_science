#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:55:00 2019

@author: sahand
"""

import sys
import time
import gc
import collections
import json
import re
import os
import pprint
from random import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, AffinityPropagation
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from yellowbrick.cluster import KElbowVisualizer
import scipy.cluster.hierarchy as sch
from scipy import spatial,sparse,sign

from bokeh.io import push_notebook, show, output_notebook, output_file
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

import fasttext
from gensim.models import FastText as fasttext_gensim
from gensim.test.utils import get_tmpfile


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))

from sciosci.assets import keyword_assets as kw
from sciosci.assets import generic_assets as sci
from sciosci.assets import advanced_assets as aa

# Read cluster centers
cluster_centers = pd.read_csv('/home/sahand/GoogleDrive/Data/FastText doc clusters - SIP/50D/cluster_centers/agglomerative ward 1990-2004 7',index_col=0)

# Read and make keyword list
keywords = pd.read_csv('/home/sahand/GoogleDrive/Data/Author keywords - 02 Nov 2019/2017-2018 keyword frequency',names=['keyword','frequency'])
keywords = keywords[keywords['frequency']>20]
keywords_list = keywords['keyword'].values.tolist()
# Get keyword embeddings
gensim_model_address = '/home/sahand/GoogleDrive/Data/FastText Models/50D/fasttext-scopus_wos-merged-310k_docs-gensim 50D.model'
model = fasttext_gensim.load(gensim_model_address)

# Save in a dict
output_dict = {}
for token in tqdm(keywords_list[:],total=len(keywords_list[:])):
    output_dict[token] = str(model.wv[token])

