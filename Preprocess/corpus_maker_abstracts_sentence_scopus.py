#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:28:36 2019

@author: github.com/sahandv
"""
import sys
import gc
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from sciosci.assets import text_assets as kw
from sciosci.assets import keyword_dictionaries as kd
from itertools import chain

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
tqdm.pandas()
nltk.download('wordnet')
nltk.download('punkt')
# =============================================================================
# Read data and Initialize
# =============================================================================
year_from = 1900
year_to = 2020

abstract_length_min = 300 #character lower bound
abstract_length_min_w = 60 # word lower bound

MAKE_SENTENCE_CORPUS = False
MAKE_SENTENCE_CORPUS_ADVANCED_KW = False
MAKE_SENTENCE_CORPUS_ADVANCED = False
MAKE_REGULAR_CORPUS = True
GET_WORD_FREQ_IN_SENTENCE = False
PROCESS_KEYWORDS = False

stops = ['a','an','we','result','however','yet','since','previously','although','propose','proposed','this']
nltk.download('stopwords')
stop_words = list(set(stopwords.words("english")))+stops

data_path_rel = '/home/sahand/GoogleDrive/Data/Corpus/Scopus new/TITLE-ABS-KEY AND PUBYEAR BFR 2019 - relevant - cleaned - Authors fixed - sorted - 01 Aug 2019.csv'
# data_path_rel = '/home/sahand/GoogleDrive/Data/Corpus/AI 4k/scopus_4k.csv'
# data_path_rel = '/home/sahand/Downloads/AI ALL 1900-2019'
# data_path_rel = '/home/sahand/GoogleDrive/Data/Corpus/AI 300/merged - scopus_v2_relevant wos_v1_relevant - duplicate doi removed - abstract corrected - 05 Aug 2019.csv'
data_path_rel_b = '/home/sahand/GoogleDrive/Data/Corpus/Scopus new/TITLE-ABS-KEY AND PUBYEAR AFTR 2018 - with kw - dup removed - 01 Aug 2021'

data_full_relevant = pd.read_csv(data_path_rel)
data_full_relevant_b = pd.read_csv(data_path_rel_b)
data_full_relevant = data_full_relevant.append(data_full_relevant_b)
# data_full_relevant = data_full_relevant[['dc:title','authkeywords','abstract','year']]
# data_full_relevant.columns = ['TI','DE','AB','PY']
sample = data_full_relevant.sample(400)

data_full_relevant = data_full_relevant[(data_full_relevant['subtype']=='cp') | (data_full_relevant['subtype']=='ar')]


# Keyword extraction from scopus
kwords = data_full_relevant[pd.notnull(data_full_relevant['authkeywords'])]['authkeywords'].str.lower().values.tolist()
kwords = [x.split(" | ") for x in kwords]
kwords_flatten = pd.DataFrame(list(chain.from_iterable(kwords)),columns=['DE'])
kwords_flatten = kwords_flatten[pd.notnull(kwords_flatten['DE'])]
kwords_flatten = kwords_flatten[kwords_flatten['DE']!='']
kwords_flatten['DE'] = kwords_flatten['DE'].str.strip('"`-+!?_ ')
kwords_flatten_grouped = kwords_flatten['DE'].value_counts().reset_index()
kwords_flatten_grouped.columns = ['DE','count']
# kwords_flatten_grouped_q = kwords_flatten_grouped['count'].quantile(0.80,interpolation='nearest')
kwords_flatten_grouped = kwords_flatten_grouped[kwords_flatten_grouped['count']>2]
kwords_unique = set(kwords_flatten_grouped['DE'].values.tolist())
pd.DataFrame(kwords_unique).to_csv('/home/sahand/GoogleDrive/Data/Corpus/Taxonomy/AI ALL Scopus n>2',index=False)


root_dir = '/home/sahand/GoogleDrive/Data/Corpus/Scopus new/'
subdir = 'clean/' # no_lemmatization_no_stopwords
gc.collect()


data_full_relevant['PY'] = data_full_relevant['year']
data_full_relevant['AB'] = data_full_relevant['dc:description']
data_full_relevant['TI'] = data_full_relevant['dc:title']
data_full_relevant['DE'] = data_full_relevant['authkeywords']
data_full_relevant['ID'] = ''
data_full_relevant['SO'] = data_full_relevant['prism:publicationName']
data_full_relevant['id'] = data_full_relevant['dc:identifier']
# 
data_wrong = data_full_relevant[data_full_relevant['AB'].str.contains("abstract available")].index
data_wrong = list(data_wrong)
data_full_relevant = data_full_relevant.drop(data_wrong,axis=0)
# =============================================================================
# Initial Pre-Processing : 
#   Following tags requires WoS format. Change them otherwise.
# =============================================================================
data_filtered = data_full_relevant.copy()
data_filtered = data_filtered[pd.notnull(data_filtered['PY'])]

# data_filtered = data_filtered[data_filtered['PY'].astype('int')>year_from-1]
# data_filtered = data_filtered[data_filtered['PY'].astype('int')<year_to]

# Remove columns without keywords/abstract list 
# data_with_keywords = data_filtered[pd.notnull(data_filtered['DE'])]
data_with_abstract = data_filtered[pd.notnull(data_filtered['AB'])]

# Remove numbers from abstracts to eliminate decimal points and other unnecessary data
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_c(x) if pd.notnull(x) else np.nan).str.lower()
data_with_abstract['TI'] = data_with_abstract['TI'].progress_apply(lambda x: kw.find_and_remove_c(x) if pd.notnull(x) else np.nan).str.lower()
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'et al.') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'eg.') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'ie.') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'e.g.') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'i.e.') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'vs.') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'ieee') if pd.notnull(x) else np.nan)
data_with_abstract['TI'] = data_with_abstract['TI'].progress_apply(lambda x: kw.find_and_remove_term(x,'ieee') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'\\usepackage') if pd.notnull(x) else np.nan)
data_with_abstract['TI'] = data_with_abstract['TI'].progress_apply(lambda x: kw.find_and_remove_term(x,'\n',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'\n',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'λ','lambda') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'β','beta') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'η','eta') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'σ','delta') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'α','alpha') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'γ','y') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'é','e') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'š','s') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'ı','i') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'<mrow>',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'</mrow>',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'<annotation>',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'</annotation>',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'p2p','peer to peer') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'<mi>',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'</mi>',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'<mo>',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'</mo>',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'<msub>',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'</msub>',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'<semantics>',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'</semantics>',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'<math>',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'</math>',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'<sub>',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'</sub>',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'+',' plus ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'<p>',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'</p>',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'<italic>',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'</italic>',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: re.sub(r'http\S+', ' ', x)if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'xmlns:xsi=',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'xmlns=',' ') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'fig.','figure') if pd.notnull(x) else np.nan)
data_with_abstract['id'] = data_with_abstract['id'].progress_apply(lambda x: kw.find_and_remove_term(x,'SCOPUS_ID:') if pd.notnull(x) else np.nan)
sample = data_with_abstract.sample(100)

num2words = {'0':' zero ','1':' one ','2':' two ','3':' three ','4':' four ','5':' five ','6':' six ','7':' seven ','8':' eight ','9':' nine '}
num2words = {'0':' zero ','1':' one ','2':' two ','3':' three ','4':' four ','5':' five ','6':' six ','7':' seven ','8':' eight ','9':' nine '}

def replace_nums(string,dictionary,regex="(?<!\d)\d(?!\d)"):
    while True:
        try:
            index = re.search(regex, string).start()
            string_a = string[:index]
            string_b = string[index+1:]
            string_a = string_a + dictionary[string[index]]
            string = string_a+string_b
        except:
            return string
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: replace_nums(x,num2words) if pd.notnull(x) else np.nan)


# gc.collect()
abstracts = []
titles = []
ids = []
for i,row in tqdm(data_with_abstract.iterrows(),total=data_with_abstract.shape[0]):
    abstract = row['AB']
    title = row['TI']
    numbers_ab = re.findall(r"[-+]?\d*\.\d+|\d+", abstract)
    numbers_ti = re.findall(r"[-+]?\d*\.\d+|\d+", title)

    for number in numbers_ab:
        abstract = kw.find_and_remove_term(abstract,number)
    for number in numbers_ti:
        title = kw.find_and_remove_term(title,number)
    abstracts.append(abstract)
    titles.append(title)
    ids.append(row['id'])

data_with_abstract['AB'] = abstracts
data_with_abstract['TI'] = titles
data_with_abstract['id_n'] = ids

assert data_with_abstract['id'].equals(data_with_abstract['id_n']), "Oh no! id mismatch here... Please fix it!"

del  abstracts
del  titles
data_with_abstract = data_with_abstract[pd.notnull(data_with_abstract['AB'])]

# source_list = pd.DataFrame(data_with_abstract['SO'].values.tolist(),columns=['source'])
# source_list.to_csv(root_dir+subdir+str(year_from)+'-'+str(year_to-1)+' corpus sources',index=False) # Save year indices to disk for further use

# year_list = pd.DataFrame(data_with_abstract['PY'].values.tolist(),columns=['year'])
# year_list.to_csv(root_dir+subdir+str(year_from)+'-'+str(year_to-1)+' corpus years',index=False) # Save year indices to disk for further use
gc.collect()

# =============================================================================
# Clean bad data based on abstracts
# =============================================================================
long_abstracts = []
lens = []
word_len = []
percentile = 5
for i,row in tqdm(data_with_abstract.iterrows(),total=data_with_abstract.shape[0]):
    leng = len(row['AB'])
    w_leng = len(row['AB'].split())
    word_len.append([len(ab) for ab in row['AB'].split()])
    lens.append(leng)
    if leng > abstract_length_min and w_leng > abstract_length_min_w:
        long_abstracts.append(row['id'])

word_len_f = [j for sub in word_len for j in sub]
median_word_len = np.median(word_len_f)
mean_word_len = np.mean(word_len_f)

max_paragraph_len = int(np.percentile(lens, percentile)) # take Nth percentile as the sentence length threshold
data_with_abstract = data_with_abstract[data_with_abstract['id'].isin(long_abstracts)]
data_with_abstract = data_with_abstract[data_with_abstract['id']!='']
data_with_abstract = data_with_abstract[pd.notnull(data_with_abstract['AB'])]
data_with_abstract = data_with_abstract[pd.notnull(data_with_abstract['id'])]
data_with_abstract = data_with_abstract.drop(['id_n'],axis=1)

data_with_abstract = data_with_abstract.reset_index(drop=True)


# =============================================================================
# Save to disk
# =============================================================================
data_with_abstract.to_csv(root_dir+subdir+'data with abstract',index=False)

id_list = pd.DataFrame(data_with_abstract['id'].values.tolist(),columns=['id'])
id_list.to_csv(root_dir+subdir+'publication idx',index=False) # Save year indices to disk for further use

year_list = pd.DataFrame(data_with_abstract['PY'].values.tolist(),columns=['year'])
year_list.to_csv(root_dir+subdir+'corpus years',index=False) # Save year indices to disk for further use
gc.collect()

year_list.plot.hist(bins=60, alpha=0.5,figsize=(15,6))
year_list.shape

# =============================================================================
# Sentence maker
# =============================================================================
if MAKE_SENTENCE_CORPUS is True:
    thesaurus = pd.read_csv('data/thesaurus/thesaurus_for_ai_keyword_with_() (training).csv')
    thesaurus = thesaurus.fillna('')
    print("\nSentence maker and thesaurus matching. \nThis will take some time...")
    
    data_with_abstract['AB_no_c'] = data_with_abstract['AB'].apply(lambda x: kw.find_and_remove_c(x) if pd.notnull(x) else np.nan)
    sentence_corpus = []
    
    for index,row in tqdm(data_with_abstract.iterrows(),total=data_with_abstract.shape[0]):
        words = re.split('( |\\n|\.|\?|!|:|;|,|_|\[|\])',row['AB_no_c'].lower())
        new_words = []
        year = row['PY']
        flag_word_removed = False
        for w_idx,word in enumerate(words):
            if flag_word_removed is True:
                if word==' ':
                    flag_word_removed = False
                    continue
            if word in thesaurus['alt'].values.tolist():
                word_old = word
                buffer_word = word
                word = thesaurus[thesaurus['alt']==word]['original'].values.tolist()[0]
#                print("changed '",word_old,"' to '",word,"'.")
                
            new_words.append(word)
            
        row = ''.join(new_words)
        
        sentences = re.split('(\. |\? |\\n)',row)
        sentences = [i+j for i,j in zip(sentences[0::2], sentences[1::2])]
        
        for sentence_n in sentences:
            sentence_corpus.append([index,sentence_n,year])
    
    sentence_corpus = pd.DataFrame(sentence_corpus,columns=['article_index','sentence','year'])
    
    sentence_corpus.to_csv(root_dir+subdir+str(year_from)+'-'+str(year_to-1)+' corpus sentences abstract-title',index=False,header=True)

gc.collect()
# =============================================================================
# Sentence maker -- Advanced -- 
# =============================================================================
if MAKE_SENTENCE_CORPUS_ADVANCED is True:    
    data_with_abstract['TI_AB'] = data_with_abstract.TI.map(str) + ". " + data_with_abstract.AB
    data_fresh = data_with_abstract[['TI_AB','PY']].copy()
    data_fresh['TI_AB'] = data_fresh['TI_AB'].str.lower()
    
    del data_with_abstract
    gc.collect()
    
    data_tmp = data_fresh[1:10]
    data_fresh[-2:-1]

    print("\nSentence extraction")
    sentences = []
    years = []
    indices = []
    for index,row in tqdm(data_fresh.iterrows(),total=data_fresh.shape[0]):
        abstract_str = row['TI_AB']
        year = row['PY']
        abstract_sentences = re.split('\. |\? |\\n',abstract_str)
        length = len(abstract_sentences)
        
        sentences.extend(abstract_sentences)
        years.extend([year for x in range(length)])
        indices.extend([index for x in range(length)])
        
    print("\nTokenizing")
    tmp = []
    for sentence in tqdm(sentences):
        tmp.append(word_tokenize(sentence))
    sentences = tmp.copy()
    del tmp

    print("\nString pre processing for abstracts: lower and strip")
    sentences = [list(map(str.lower, x)) for x in sentences]
    sentences = [list(map(str.strip, x)) for x in sentences]
    
    tmp = []
    print("\nString pre processing for abstracts: lemmatize and stop word removal")
    for string_list in tqdm(sentences, total=len(sentences)):
        tmp_list = [kw.string_pre_processing(x,stemming_method='None',lemmatization='DEF',stop_word_removal=True,stop_words_extra=stops,verbose=False,download_nltk=False) for x in string_list]
        tmp.append(tmp_list)
    sentences = tmp.copy()
    del tmp
    gc.collect()
    
    tmp = []
    print("\nString pre processing for abstracts: null word removal")
    for string_list in tqdm(sentences, total=len(sentences)):
        tmp.append([x for x in string_list if x!=''])
    sentences = tmp.copy()
    del tmp
    
    print("\nThesaurus matching")
    sentences = kw.thesaurus_matching(sentences,thesaurus_file='data/thesaurus/thesaurus_for_ai_keyword_with_() (training).csv')
    
    print("\nStitiching tokens")
    tmp = []
    for words in tqdm(sentences, total=len(sentences)):
        tmp.append(' '.join(words))
    sentences = tmp.copy()
    del tmp
    
    print("\nGB to US")
    tmp = []
    for sentence in tqdm(sentences, total=len(sentences)):
        tmp.append(kw.replace_british_american(sentence,kd.gb2us))
    sentences = tmp.copy()
    del tmp
    
    sentence_df = pd.DataFrame(indices,columns=['article_index'])
    sentence_df['sentence'] = sentences
    sentence_df['year'] = years
    sentence_df.to_csv(root_dir+subdir+'corpus sentences abstract-title',index=False,header=True)
    
# =============================================================================
# Keyword Extractor
# =============================================================================
if MAKE_SENTENCE_CORPUS_ADVANCED_KW is True:    
    data_with_abstract['TI_AB'] = data_with_abstract.AB
    data_fresh = data_with_abstract[['TI_AB','PY']].copy()
    data_fresh['TI_AB'] = data_fresh['TI_AB'].str.lower()
    
    del data_with_abstract
    gc.collect()
    
    data_tmp = data_fresh[1:10]
    data_fresh[-2:-1]

    print("\nSentence extraction")
    sentences = []
    years = []
    indices = []
    for index,row in tqdm(data_fresh.iterrows(),total=data_fresh.shape[0]):
        abstract_str = row['TI_AB']
        year = row['PY']
        abstract_sentences = re.split('\\n',abstract_str)
        length = len(abstract_sentences)
        
        sentences.extend(abstract_sentences)
        years.extend([year for x in range(length)])
        indices.extend([index for x in range(length)])
        
    print("\nTokenizing")
    tmp = []
    for sentence in tqdm(sentences):
        tmp.append(word_tokenize(sentence))
    sentences = tmp.copy()
    del tmp

    print("\nString pre processing for abstracts: lower and strip")
    sentences = [list(map(str.lower, x)) for x in sentences]
    sentences = [list(map(str.strip, x)) for x in sentences]
    
    tmp = []
    print("\nString pre processing for abstracts: lemmatize and stop word removal")
    for string_list in tqdm(sentences, total=len(sentences)):
        tmp_list = [kw.string_pre_processing(x,stemming_method='None',lemmatization='DEF',stop_word_removal=True,stop_words_extra=stops,verbose=False,download_nltk=False) for x in string_list]
        tmp.append(tmp_list)
    sentences = tmp.copy()
    del tmp
    gc.collect()
        
    tmp = []
    print("\nString pre processing ")
    for string_list in tqdm(sentences, total=len(sentences)):
        string_tmp = []
        for token in string_list:
            if token == '':
                string_tmp.append(' | ')
            else:
                string_tmp.append(token)
        tmp.append(string_tmp)
    sentences = tmp.copy()
    del tmp
    
    tmp = []
    print("\nString pre processing for abstracts: null word removal")
    for string_list in tqdm(sentences, total=len(sentences)):
        tmp.append([x for x in string_list if x!=''])
    sentences = tmp.copy()
    del tmp
    
    print("\nThesaurus matching")
    sentences = kw.thesaurus_matching(sentences,thesaurus_file='data/thesaurus/thesaurus_for_ai_keyword_with_() (testing).csv')
    
    print("\nStitiching tokens")
    tmp = []
    for words in tqdm(sentences, total=len(sentences)):
        tmp.append(' '.join(words))
    sentences = tmp.copy()
    del tmp
    
    print("\nGB to US")
    tmp = []
    for sentence in tqdm(sentences, total=len(sentences)):
        tmp.append(kw.replace_british_american(sentence,kd.gb2us))
    sentences = tmp.copy()
    del tmp
    
    sentence_df = pd.DataFrame(indices,columns=['article_index'])
    sentence_df['sentence'] = sentences
    sentence_df['year'] = years
    sentence_df.to_csv(root_dir+subdir+str(year_from)+'-'+str(year_to-1)+' corpus sentences abstract-title',index=False,header=True)
    

if MAKE_REGULAR_CORPUS is False:
    sys.exit('Did not continue to create normal corpus. If you want a corpus, set it to True at init section.')
# =============================================================================
#   Get word frequency in sentence corpus -- OPTIONAL
# =============================================================================
if GET_WORD_FREQ_IN_SENTENCE is True:
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    
    file = root_dir+subdir+str(year_from)+'-'+str(year_to-1)+' corpus abstract-title'#'/home/sahand/GoogleDrive/Data/corpus/AI ALL/1900-2019 corpus sentences abstract-title'
    file = pd.read_csv(file)
    size = 500000
    unique = []
    for data_start_point in tqdm(np.arange(0,file.shape[0],size)):
        if data_start_point+size<file.shape[0]:
            end_point = data_start_point+size
        else:
            end_point = file.shape[0]-1
    #    print(data_start_point,end_point)
        str_split = list(file.sentence[data_start_point:end_point].str.split())
        str_flat = pd.DataFrame([item for sublist in str_split for item in sublist])
        str_flat.columns = ['words']
        str_flat.head()
    
        unique = unique+list(str_flat.words.unique())
    
    unique = pd.DataFrame(unique)
    unique.columns = ['words']
    unique = list(unique.words.unique())
    len(unique)

# =============================================================================
# Tokenize (Author Keywords and Abstracts+Titles)
# =============================================================================
abstracts = []
keywords = []
keywords_index = []
abstracts_pure = []
for index,paper in tqdm(data_with_abstract.iterrows(),total=data_with_abstract.shape[0]):
    keywords_str = paper['DE']
    keywords_index_str = paper['ID']
    abstract_str = paper['AB']
    title_str = paper['TI']
    abstract_dic = word_tokenize(title_str+' '+abstract_str)
    abstract_dic_pure = abstract_dic.copy()
    if pd.notnull(paper['DE']):
        keywords_dic = word_tokenize(keywords_str)
        keywords.append(keywords_str.split(';'))
        abstract_dic.extend(keywords_dic)
    else:
        keywords.append([])
    if pd.notnull(paper['ID']):
        keywords_index.append(keywords_index_str.split(';'))
    else:
        keywords_index.append([])
    abstracts.append(abstract_dic)
    abstracts_pure.append(abstract_dic_pure)

# Add to main df. Not necessary
data_with_abstract['AB_split'] = abstracts_pure 
data_with_abstract['AB_KW_split'] = abstracts

# =============================================================================
# Strip and lowe case 
# =============================================================================
abstracts_pure = [list(map(str.strip, x)) for x in abstracts_pure]
abstracts_pure = [list(map(str.lower, x)) for x in abstracts_pure]

abstracts = [list(map(str.strip, x)) for x in abstracts]
abstracts = [list(map(str.lower, x)) for x in abstracts]

keywords = [list(map(str.strip, x)) for x in keywords]
keywords = [list(map(str.lower, x)) for x in keywords]

keywords_index = [list(map(str.strip, x)) for x in keywords_index]
keywords_index = [list(map(str.lower, x)) for x in keywords_index]
# =============================================================================
# Pre Process 
# =============================================================================
tmp_data = []
print("\nString pre processing for ababstracts_purestracts")
for string_list in tqdm(abstracts, total=len(abstracts)):
    tmp_list = [kw.string_pre_processing(x,stemming_method='None',lemmatization=False,stop_word_removal=False,stop_words_extra=stops,verbose=False,download_nltk=False) for x in string_list]
    tmp_data.append(tmp_list)
abstracts = tmp_data.copy()
del tmp_data

tmp_data = []
for string_list in tqdm(abstracts_pure, total=len(abstracts_pure)):
    tmp_list = [kw.string_pre_processing(x,stemming_method='None',lemmatization='DEF',stop_word_removal=True,stop_words_extra=stops,verbose=False,download_nltk=False) for x in string_list]
    tmp_data.append(tmp_list)
abstracts_pure = tmp_data.copy()
del tmp_data

if PROCESS_KEYWORDS is True:
    print("\nString pre processing for keywords")
    tmp_data = []
    for string_list in tqdm(keywords, total=len(keywords)):
        tmp_list = []
        for string in string_list:
            tmp_sub_list = string.split()
            tmp_list.append(' '.join([kw.string_pre_processing(x,stemming_method='None',lemmatization=False,stop_word_removal=True,stop_words_extra=stops,verbose=False,download_nltk=False) for x in tmp_sub_list]))
        tmp_data.append(tmp_list)
    keywords = tmp_data.copy()
    del tmp_data
    
    tmp_data = []
    for string_list in tqdm(keywords_index, total=len(keywords_index)):
        tmp_list = []
        for string in string_list:
            tmp_sub_list = string.split()
            tmp_list.append(' '.join([kw.string_pre_processing(x,stemming_method='None',lemmatization=False,stop_word_removal=True,stop_words_extra=stops,verbose=False,download_nltk=False) for x in tmp_sub_list]))
        tmp_data.append(tmp_list)
    keywords_index = tmp_data.copy()
    del tmp_data

#tmp_data = []
#for string_list in tqdm(keywords, total=len(keywords)):
#    tmp_list = []
#    for sub_string_list in string_list:
#        tmp_list.append(' '.join(sub_string_list))
#    tmp_data.append(tmp_list)
#keywords = tmp_data.copy()
#del tmp_data

# =============================================================================
# Clean-up dead words
# =============================================================================
tmp_data = []
for string_list in tqdm(abstracts, total=len(abstracts)):
    tmp_data.append([x for x in string_list if x!=''])
abstracts = tmp_data.copy()
del tmp_data

tmp_data = []
for string_list in tqdm(abstracts_pure, total=len(abstracts_pure)):
    tmp_data.append([x for x in string_list if x!=''])
abstracts_pure = tmp_data.copy()
del tmp_data

tmp_data = []
for string_list in tqdm(keywords, total=len(keywords)):
    tmp_data.append([x for x in string_list if x!=''])
keywords = tmp_data.copy()
del tmp_data

tmp_data = []
for string_list in tqdm(keywords_index, total=len(keywords_index)):
    tmp_data.append([x for x in string_list if x!=''])
keywords_index = tmp_data.copy()
del tmp_data
# =============================================================================
# Break-down abstracts again
# =============================================================================
tmp_data = []
for abstract in tqdm(abstracts):
    words = []
    for word in abstract:
        words = words+word.split()
    tmp_data.append(words)
abstracts = tmp_data.copy()
del tmp_data

tmp_data = []
for abstract in tqdm(abstracts_pure):
    words = []
    for word in abstract:
        words = words+word.split()
    tmp_data.append(words)
abstracts_pure = tmp_data.copy()
del tmp_data

# =============================================================================
# Thesaurus matching
# =============================================================================
print("\nThesaurus matching")

abstracts_backup = abstracts.copy()
abstracts_pure_backup = abstracts_pure.copy()
keywords_backup = keywords.copy()
keywords_index_backup = keywords_index.copy()

abstracts = abstracts_backup.copy()
abstracts_pure = abstracts_pure_backup.copy()
keywords = keywords_backup.copy()
keywords_index = keywords_index_backup.copy()

abstracts = kw.thesaurus_matching(abstracts,thesaurus_file='data/thesaurus/thesaurus_for_ai_keyword_with_() (testing).csv')
abstracts_pure = kw.thesaurus_matching(abstracts_pure,thesaurus_file='data/thesaurus/thesaurus_for_ai_keyword_with_() (testing).csv')
if PROCESS_KEYWORDS is True:
    keywords = kw.thesaurus_matching(keywords)
    keywords_index = kw.thesaurus_matching(keywords_index)

# =============================================================================
# Term to string corpus for co-word analysis
# =============================================================================
print("\nTerm to string corpus for co-word analysis")
corpus_abstract = []
for words in tqdm(abstracts, total=len(abstracts)):
    corpus_abstract.append(' '.join(words))

corpus_abstract_pure = []
for words in tqdm(abstracts_pure, total=len(abstracts_pure)):
    corpus_abstract_pure.append(' '.join(words))

corpus_keywords = []
for words in tqdm(keywords, total=len(keywords)):
    corpus_keywords.append(';'.join(words))
    
corpus_keywords_index = []
for words in tqdm(keywords_index, total=len(keywords_index)):
    corpus_keywords_index.append(';'.join(words))


# =============================================================================
# Remove substrings : 
#   be careful with this one! It might remove parts of a string or half of a word
# =============================================================================
thesaurus = pd.read_csv('data/thesaurus/to_remove.csv')
thesaurus['alt'] = ''
thesaurus = thesaurus.values.tolist()
print("\nRemoving substrings")

corpus_abstract_tr = []
for paragraph in tqdm(corpus_abstract, total=len(corpus_abstract)):
    paragraph = kw.filter_string(paragraph,thesaurus)
    corpus_abstract_tr.append(paragraph)

corpus_abstract_pure_tr = []
for paragraph in tqdm(corpus_abstract_pure, total=len(corpus_abstract_pure)):
    paragraph = kw.filter_string(paragraph,thesaurus)
    corpus_abstract_pure_tr.append(paragraph)

corpus_keywords_tr = []
for paragraph in tqdm(corpus_keywords, total=len(corpus_keywords)):
    paragraph = kw.filter_string(paragraph,thesaurus)
    corpus_keywords_tr.append(paragraph)
    
corpus_keywords_index_tr = []
for paragraph in tqdm(corpus_keywords_index, total=len(corpus_keywords_index)):
    paragraph = kw.filter_string(paragraph,thesaurus)
    corpus_keywords_index_tr.append(paragraph)
    
# =============================================================================
# Final clean-up (double space and leading space)
# =============================================================================
tmp_data = []
for paragraph in tqdm(corpus_abstract, total=len(corpus_abstract)):
    paragraph = ' '.join(paragraph.split())
    tmp_data.append(paragraph)
corpus_abstract = tmp_data.copy()
del tmp_data

tmp_data = []
for paragraph in tqdm(corpus_abstract_tr, total=len(corpus_abstract_tr)):
    paragraph = ' '.join(paragraph.split())
    tmp_data.append(paragraph)
corpus_abstract_tr = tmp_data.copy()
del tmp_data

tmp_data = []
for paragraph in tqdm(corpus_abstract_pure, total=len(corpus_abstract_pure)):
    paragraph = ' '.join(paragraph.split())
    tmp_data.append(paragraph)
corpus_abstract_pure = tmp_data.copy()
del tmp_data

tmp_data = []
for paragraph in tqdm(corpus_abstract_pure_tr, total=len(corpus_abstract_pure_tr)):
    paragraph = ' '.join(paragraph.split())
    tmp_data.append(paragraph)
corpus_abstract_pure_tr = tmp_data.copy()
del tmp_data

tmp_data = []
for paragraph in tqdm(corpus_keywords, total=len(corpus_keywords)):
    paragraph = ' '.join(paragraph.split(' '))
    paragraph = ';'.join(paragraph.split(';'))
    tmp_data.append(paragraph)
corpus_keywords = tmp_data.copy()
del tmp_data

tmp_data = []
for paragraph in tqdm(corpus_keywords_tr, total=len(corpus_keywords_tr)):
    paragraph = ' '.join(paragraph.split(' '))
    paragraph = ';'.join(paragraph.split(';'))
    tmp_data.append(paragraph)
corpus_keywords_tr = tmp_data.copy()
del tmp_data
tmp_data = []
for paragraph in tqdm(corpus_keywords_index, total=len(corpus_keywords_index)):
    paragraph = ' '.join(paragraph.split(' '))
    paragraph = ';'.join(paragraph.split(';'))
    tmp_data.append(paragraph)
corpus_keywords_index = tmp_data.copy()
del tmp_data

tmp_data = []
for paragraph in tqdm(corpus_keywords_index_tr, total=len(corpus_keywords_index_tr)):
    paragraph = ' '.join(paragraph.split(' '))
    paragraph = ';'.join(paragraph.split(';'))
    tmp_data.append(paragraph)
corpus_keywords_index_tr = tmp_data.copy()
del tmp_data

# =============================================================================
# Write to disk
# =============================================================================
corpus_abstract = pd.DataFrame(corpus_abstract,columns=['words'])
corpus_abstract_tr = pd.DataFrame(corpus_abstract_tr,columns=['words'])
corpus_abstract_pure = pd.DataFrame(corpus_abstract_pure,columns=['words'])
corpus_abstract_pure_tr = pd.DataFrame(corpus_abstract_pure_tr,columns=['words'])
corpus_keywords = pd.DataFrame(corpus_keywords,columns=['words'])
corpus_keywords_tr = pd.DataFrame(corpus_keywords_tr,columns=['words'])
corpus_keywords_index = pd.DataFrame(corpus_keywords_index,columns=['words'])
corpus_keywords_index_tr = pd.DataFrame(corpus_keywords_index_tr,columns=['words'])

# corpus_abstract.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' abstract_title_keys',index=False,header=False)
# corpus_abstract_tr.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' abstract_title_keys-terms_removed' ,index=False,header=False)
corpus_abstract_pure.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' abstract_title',index=False,header=False)
# corpus_abstract_pure_tr.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' abstract_title-terms_removed',index=False,header=False)
# corpus_keywords.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' keywords',index=False,header=False)
# corpus_keywords_tr.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' keywords-terms_removed',index=False,header=False)
# corpus_keywords_index.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' keywords_index',index=False,header=False)
# corpus_keywords_index_tr.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' keywords_index-terms_removed',index=False,header=False)
