#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 17:34:51 2020

@author: github.com/sahandv

Pro tip: use all data for better results
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

period = '2017-2018'
corpus = pd.read_csv('/home/sahand/GoogleDrive/Data/corpus/improved_copyr_thesaurus/'+period+' corpus abstract-title',header=None)
corpus.columns = ['abstracts']
corpus = corpus.fillna('')
corpus_list = corpus['abstracts'].tolist()

#pd.DataFrame(corpus_list).to_csv('/home/sahand/GoogleDrive/Data/corpus/improved_copyr_thesaurus/_1990-2018 corpus abstract-title',header=False,index=False)
# =============================================================================
# Pre-process
# =============================================================================
print('\nPre-processing '+period+' ...')
corpus_list_p = [kw.string_pre_processing(x,stemming_method='None',lemmatization=True,stop_word_removal=True,stop_words_extra=stops,verbose=False,download_nltk=False) for x in corpus_list]

# =============================================================================
# Process : Gensim method
# =============================================================================
print('\nProcessing...')
data_words = [re.split('\ |,|\(|\)|:|;|\[|\]|\.|\?|\!',abst) for abst in corpus_list_p]
bigram = gensim.models.Phrases(data_words, min_count=100, threshold=5) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=10)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

print("\nForming dictionary...")
data_words_with_trigrams = make_trigrams(data_words)

# Make sentence corpus
print("\nMaking new corpus with n-grams...")
data_words_with_trigrams_corpus = [' '.join(x) for x in data_words_with_trigrams]

## =============================================================================
## Process : scikit learn method
## =============================================================================
#def get_abstract_keywords(corpus):
#    cv=CountVectorizer(max_df=0.999,min_df=20,stop_words=stop_words, ngram_range=(1,3))
#    X=cv.fit_transform(corpus)
#    # get feature names
#    feature_names=cv.get_feature_names()
#    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
#    tfidf_transformer.fit(X)
#    keywords_tfidf = []
#    keywords_sorted = []
#    for doc in tqdm(corpus,total=len(corpus)):
#        tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
#        sorted_items=kw.sort_coo(tf_idf_vector.tocoo())
#        keywords_sorted.append(sorted_items)
#        keywords_tfidf.append(kw.extract_topn_from_vector(feature_names,sorted_items,len(feature_names)))
#    return keywords_tfidf
#
#kws = get_abstract_keywords(corpus_list)

# =============================================================================
# Write to disk
# =============================================================================
print('\nWiting to disk...')
pd.DataFrame(data_words_with_trigrams_corpus).to_csv('/home/sahand/GoogleDrive/Data/corpus/improved_copyr_thesaurus/n-grams/'+period+' corpus abstract-title',header=False,index=False)
#start = len(corpus_list)+start #data_words_with_trigrams_corpus[start:len(corpus_list)+start]
print('\nDone!')