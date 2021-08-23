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

path = '/home/sahand/GoogleDrive/Data/'
# data_address =  path+"Corpus/AI 4k/copyr_deflem_stopword_removed_thesaurus May 28/1990-2019/1990-2019 abstract_title"
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

#%%# ==========================================================================
# From taxonomy
# 
# The issue is that, too many possible words are out there in kw list. Like, "mind" or "eye". These are correct. But out of context. 
# We have to bee too specific if we wamt to rely om author kerywords, like the kw from 4 AI journals.
# =============================================================================
import numpy as np
import pandas as pd
from tqdm import tqdm
from sciosci.assets import text_assets as kw
from sciosci.assets import keyword_dictionaries as kd
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from multiprocessing import Pool

stops = ['a','an','we','result','however','yet','since','previously','although','propose','proposed','this','...']
stop_words = list(set(stopwords.words("english")))+stops

path = '/home/sahand/GoogleDrive/Data/'

keywords = list(set(pd.read_csv(path+'Corpus/Taxonomy/TAI Taxonomy.csv',sep='===',names=['keyword'])['keyword'].values.tolist()))
keywords = keywords+list(set(pd.read_csv(path+'Corpus/Taxonomy/1980-2019 300k n-gram author keyword taxonomy - 95percentile and cleaned.csv')['keywords'].values.tolist()))

keywords = [x for x in keywords if len(x)>2]
keywords = [kw.replace_british_american(strip_multiple_whitespaces(kw.replace_british_american(strip_multiple_whitespaces(keyword),kd.gb2us)),kd.gb2us) for keyword in keywords]
keywords = [k.strip().lower() for k in keywords]
keywords = np.array(keywords)

pub_idx = pd.read_csv(path+'Corpus/Dimensions AI unlimited citations/clean/publication idx',names=['id'])[:]
abstracts = pd.read_csv(path+'Corpus/Dimensions AI unlimited citations/clean/abstract_title pure US',names=['abstract'])[:]
idx = abstracts.index
abstracts = abstracts['abstract'].values.tolist()
# pd.DataFrame(keywords).to_csv(path+'Corpus/Taxonomy/AI kw merged US',index=False,header=False)
# =============================================================================
# abstract = word_tokenize(abstract)
# abstract = [word for word in abstract if not word in stop_words] 
# 
# extraction = [word for word in abstract if word in keywords] 
# matches = [keyword in abstract for keyword in keywords] 
# selection = keywords[matches]
# =============================================================================
pool = Pool()

def extract(abstracts):
    pubkeywords = []
    errors = []
    for i,abstract in tqdm(enumerate(abstracts),total=len(abstracts)):
        try:
            pubkeywords.append([x for x in keywords if x in abstract])
        except:
            pubkeywords.append([])
            errors.append(i)
    print('errors:'+str(errors))
    return pubkeywords,errors

extracted,errors = extract(abstracts)
extracted = list(extracted)
extracted_df = [str(list(row))[1:-1] for row in extracted]
extracted_df = pd.DataFrame(extracted_df)
extracted_df.index = idx
extracted_df.to_csv(path+'Corpus/Dimensions AI unlimited citations/clean/keyword US p1',header=False)


# =============================================================================
# # concat multiple parts
# =============================================================================
extracted_df_b = pd.read_csv(path+'Corpus/Dimensions AI unlimited citations/clean/keyword US p2',index_col=0,header=None)
extracted_df_b.columns = [0]
extracted_df = extracted_df.append(extracted_df_b)
pub_idx['data'] = extracted_df[0]
pub_idx.to_csv(path+'Corpus/Dimensions AI unlimited citations/clean/keyword US',index=False)

