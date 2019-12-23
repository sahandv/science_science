#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 16:31:24 2019

@author: github.com/sahandv
"""

from os import path
import re
import json

from sciosci import scopus_helper as sh
from sciosci.assets import generic_assets as ga

# =============================================================================
# Initialization of the package
# =============================================================================
sh.scopus_initialize()
ga.general_initialize()

from sciosci.assets import keyword_assets as kw
from sciosci.assets import generic_assets as sci

import collections
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from scipy import spatial

from sklearn.feature_extraction.text import TfidfTransformer , TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import NMF
from sklearn.model_selection import GridSearchCV

import gensim
from gensim.models import CoherenceModel, LdaModel, HdpModel
from gensim.models.wrappers import LdaVowpalWabbit, LdaMallet
from gensim.corpora import Dictionary
import gensim.corpora as corpora

import nltk
from nltk.corpus import stopwords


# =============================================================================
# Init
# =============================================================================
sns.set_style('whitegrid')
stops = ['a','an','we','result','however','yet','since','previously','although','propose','proposed','e_g','method',
         'published_elsevier','b','v','problem','paper','approach','within','with','by','via','way','t','case','issue','level','area','system',
         'work','discussed','seen','put','usually','take','make','author','versus','enables','result','research','design','based','al']
nltk.download('stopwords')
stop_words = list(set(stopwords.words("english")))+stops
np.random.seed(50)

# =============================================================================
# Gensim n-gram corpus maker (collection and phrase detector)
# =============================================================================
import warnings
warnings.filterwarnings('ignore')

save_new_corpus = True
folds = 5
start=2
limit=40
step=2

year_from = 1990
year_to = 2005
period = str(year_from)+'-'+str(year_to-1)

data_path_rel = '/home/sahand/GoogleDrive/Data/Relevant Results _ DOI duplication - scopus keywords - document types - 31 july.csv'
data_full_relevant_wos = pd.read_csv(data_path_rel)
data_filtered = data_full_relevant_wos.copy()
data_filtered = data_filtered[data_filtered['PY'].astype('int')>year_from-1]
data_filtered = data_filtered[data_filtered['PY'].astype('int')<year_to]

corpus = pd.read_csv('/home/sahand/GoogleDrive/Data/corpus/improved_copyr_abstract-sentences_separated/'+period+' corpus sentences abstract')
corpus = corpus[pd.notnull(corpus['sentence'])]
corpus_sentences = corpus['sentence'].tolist()
corpus_article_indices = corpus['article_index'].tolist()

print("\nTokenizing docs for trigram-generation\n")
data_words = [re.split('\ |,|\(|\)|:|;|\[|\]|\.|\?|\!',abst) for abst in corpus_sentences]

print("\nForming trigrams\n")
# =============================================================================
# # Build the bigram and trigram models
# =============================================================================
bigram = gensim.models.Phrases(data_words, min_count=7, threshold=70) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=30)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

print("\nForming dictionary\n")
data_words_with_trigrams = make_trigrams(data_words)

# Make sentence corpus
print("\nMaking new corpus with n-grams\n")
data_words_with_trigrams_corpus = [' '.join(x) for x in data_words_with_trigrams]

# =============================================================================
# # Make abstract corpus
# =============================================================================
abstract_indices_articles = []
abstracts = []
buffer_idx = None
buffer_abstract = ''
first_flag = True
for idx,idx_article in tqdm(enumerate(corpus_article_indices),total=len(corpus_article_indices)):
#    if data_words_with_trigrams_corpus[idx] != '':
    if idx_article == buffer_idx:
        # this is the same article
        buffer_abstract = buffer_abstract+data_words_with_trigrams_corpus[idx]+'. '
    else:
        # this is a new article
        if first_flag is False:
            abstracts.append(buffer_abstract)
            abstract_indices_articles.append(buffer_idx)
        buffer_abstract = data_words_with_trigrams_corpus[idx]+'. '
    
    first_flag = False
    buffer_idx = idx_article

# =============================================================================
# # Write to disk
# =============================================================================
data_with_meta_data = data_filtered.loc[abstract_indices_articles,:]
data_with_meta_data['processed_abstracts'] = abstracts
data_with_meta_data.to_csv('/home/sahand/GoogleDrive/Data/corpus/improved_copyr_abstract-sentences_separated/meta/'+period+'  meta and data.csv',header=True,index=True)

data_words_with_trigrams_corpus = pd.DataFrame(data_words_with_trigrams_corpus)
data_words_with_trigrams_corpus.to_csv('/home/sahand/GoogleDrive/Data/corpus/improved_copyr_abstract-sentences_separated/with n-grams/'+period+'  corpus sentences abstract - with n-grams',header=False,index=False)


# =============================================================================
# # TF-IDF
# =============================================================================
print("TF-IDF calculation")
max_features = 1000000
cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=max_features, ngram_range=(1,1))
X=cv.fit_transform(abstracts)
keys = list(cv.vocabulary_.keys())[:1000]
feature_names=cv.get_feature_names()

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(X)

keywords_tfidf = []
keywords_sorted = []
for doc in tqdm(abstracts,total=len(abstracts)):
    tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
    sorted_items=kw.sort_coo(tf_idf_vector.tocoo())
    keywords_sorted.append(sorted_items)
    keywords_tfidf.append(kw.extract_topn_from_vector(feature_names,sorted_items,20))

with open('/home/sahand/GoogleDrive/Data/corpus/improved_copyr_abstract-sentences_separated/TFIDF/'+period+'tf-idf-keywords-concatenated.json', 'w') as json_file:
    json.dump(keywords_tfidf, json_file)
