#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 17:34:51 2020

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

period = '1990-2004'
corpus = pd.read_csv('/home/sahand/GoogleDrive/Data/corpus/improved_copyr_thesaurus/'+period+' corpus abstract-title',header=None)
corpus.columns = ['abstracts']
corpus = corpus.fillna('')
corpus_list = corpus['abstracts'].tolist()
data_words = [re.split('\ |,|\(|\)|:|;|\[|\]|\.|\?|\!',abst) for abst in corpus_list]
bigram = gensim.models.Phrases(data_words, min_count=7, threshold=70) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=30)

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
# Write to disk
# =============================================================================

pd.DataFrame(data_words_with_trigrams_corpus).to_csv('/home/sahand/GoogleDrive/Data/corpus/improved_copyr_thesaurus/n-grams/'+period+' corpus abstract-title',header=False,index=False)