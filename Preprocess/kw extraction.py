#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 00:13:06 2020

@author: sahand
"""

from rake_nltk import Rake
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
st = set(stopwords.words('english'))

path = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/'
data_address =  path+"Corpus/AI 4k/copyr_deflem_stopword_removed_thesaurus May 28/1990-2019/1990-2019 abstract_title"
df1 = pd.read_csv(data_address,names=['abstract'])
labels = pd.read_csv(path+'Corpus/AI 4k/embeddings/clustering/k10/Doc2Vec patent_wos_ai corpus DEC 200,500,10 k10 labels')
df1['label'] = labels['label']
corpus = []
for cluster in df1.groupby('label').groups:
    corpus.append( ' '.join(df1[df1['label']==cluster]['abstract'].values.tolist()))

# =============================================================================
# TFIDF
# =============================================================================
all_keys = []
all_keyscores = []
for cor in corpus:
    text = cor
    text = text.replace('.',' ')
    text = re.sub(r'\s+',' ',re.sub(r'[^\w \s]','',text) ).lower()
    corpus_n = re.split('chapter \d+',text)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus_n)
    names = vectorizer.get_feature_names()
    data = vectors.todense().tolist()
    # Create a dataframe with the results
    df = pd.DataFrame(data, columns=names)
    df = df[filter(lambda x: x not in list(st) , df.columns)]
    N = 10;
    keys = []
    keyscores = []
    for i in df.iterrows():
        keyscores.append(i[1].sort_values(ascending=False)[:N].values.tolist())
        keys.append(list(i[1].sort_values(ascending=False)[:N].index))
    all_keys.append(keys)
    all_keyscores.append(keyscores)
all_keys_df = pd.DataFrame(np.array(all_keys).squeeze())
all_keys_df.to_csv()

# =============================================================================
# Rake -- won't work with long text
# =============================================================================
r = Rake()
r.extract_keywords_from_text(corpus[0])
r.get_ranked_phrases()

# =============================================================================
# Embedding
# =============================================================================

