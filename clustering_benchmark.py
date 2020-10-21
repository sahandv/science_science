#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:10:03 2020

@author: sahand
"""
import sys
import time
import gc
import collections
import json
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, AffinityPropagation
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfTransformer , TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sciosci.assets import text_assets as ta

# =============================================================================
# Load data and init
# =============================================================================
datapath = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/'

file_address =  datapath+"Corpus/KPRIS/embeddings/deflemm/Doc2Vec patent corpus"
vectors = pd.read_csv(file_address)
file_address =  datapath+"Corpus/KPRIS/labels"
labels = pd.read_csv(file_address,names=['label'])
n_clusters = 5

labels_task_1 = labels[(labels['label']=='car') | (labels['label']=='memory')]
vectors_task_1 = vectors.iloc[labels_task_1.index]
n_clusters_task_1 = 2
# =============================================================================
# K-means
# =============================================================================


# =============================================================================
# Agglomerative
# =============================================================================



# =============================================================================
# Deep
# =============================================================================


