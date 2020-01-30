#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:28:36 2019

@author: github.com/sahandv
"""
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from sciosci.assets import keyword_assets as kw

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
tqdm.pandas()

# =============================================================================
# Read data and Initialize
# =============================================================================
year_from = 1900
year_to = 2020

make_sentence_corpus = True
make_normal_corpus = False

stops = ['a','an','we','result','however','yet','since','previously','although','propose','proposed','this']
nltk.download('stopwords')
stop_words = list(set(stopwords.words("english")))+stops

#data_path_rel = '/mnt/6016589416586D52/Users/z5204044/Documents/Dataset/WoS/Relevant Results _ DOI duplication - scopus keywords - document types - 31 july.csv'
data_path_rel = '/mnt/6016589416586D52/Users/z5204044/Documents/Dataset/Scopus/AI/ALL/processed/AI ALL 1900-2019 -reformat'
data_full_relevant = pd.read_csv(data_path_rel)

root_dir = '/mnt/6016589416586D52/Users/z5204044/Documents/Dataset/Scopus/AI/ALL/processed/'
subdir = 'improved_copyr_lemmatized_stopword_removed_thesaurus_sep/' # no_lemmatization_no_stopwords


# =============================================================================
# Initial Pre-Processing : 
#   Following tags requires WoS format. Change them otherwise.
# =============================================================================
data_filtered = data_full_relevant.copy()

data_filtered = data_filtered[data_filtered['PY'].astype('int')>year_from-1]
data_filtered = data_filtered[data_filtered['PY'].astype('int')<year_to]

# Remove columns without keywords/abstract list 
data_with_keywords = data_filtered[pd.notnull(data_filtered['DE'])]
data_with_abstract = data_filtered[pd.notnull(data_filtered['AB'])]

data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_c(x) if pd.notnull(x) else np.nan)

year_list = pd.DataFrame(data_with_abstract['PY'].values.tolist(),columns=['year'])
year_list.to_csv(root_dir+str(year_from)+'-'+str(year_to-1)+' corpus years',index=False) # Save year indices to disk for further use
# =============================================================================
# Sentence maker
# =============================================================================
if make_sentence_corpus is True:
    thesaurus = pd.read_csv('thesaurus_for_ai_keyword_with_().csv')
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

if make_normal_corpus is False:
    sys.exit('Did not continue to create normal corpus. If you want a corpus, set it to True at init section.')

# =============================================================================
# Tokenize (Author Keywords and Abstracts+Titles)
# =============================================================================
abstracts = []
abstracts_pure = []
for index,paper in tqdm(data_with_abstract.iterrows(),total=data_with_abstract.shape[0]):
    keywords_str = paper['DE']
    abstract_str = paper['AB']
    title_str = paper['TI']
    abstract_dic = word_tokenize(title_str+' '+abstract_str)
    abstract_dic_pure = abstract_dic.copy()
    if pd.notnull(paper['DE']):
        keywords_dic = word_tokenize(keywords_str)
        abstract_dic.extend(keywords_dic) 
    abstracts.append(abstract_dic)
    abstracts_pure.append(abstract_dic_pure)

# Add to main df. Not necessary
data_with_abstract['AB_split'] = abstracts_pure 
data_with_abstract['AB_KW_split'] = abstracts

# =============================================================================
# Strip and lowe case keywords
# =============================================================================
abstracts_pure = [list(map(str.strip, x)) for x in abstracts_pure]
abstracts_pure = [list(map(str.lower, x)) for x in abstracts_pure]

abstracts = [list(map(str.strip, x)) for x in abstracts]
abstracts = [list(map(str.lower, x)) for x in abstracts]

# =============================================================================
# Pre Process Keywords
# =============================================================================
tmp_data = []
print("\nString pre processing for abstracts")
for string_list in tqdm(abstracts, total=len(abstracts)):
    tmp_list = [kw.string_pre_processing(x,stemming_method='None',lemmatization=False,stop_word_removal=False,stop_words_extra=stops,verbose=False,download_nltk=False) for x in string_list]
    tmp_data.append(tmp_list)
abstracts = tmp_data.copy()
del tmp_data

tmp_data = []
for string_list in tqdm(abstracts_pure, total=len(abstracts_pure)):
    tmp_list = [kw.string_pre_processing(x,stemming_method='None',lemmatization=False,stop_word_removal=False,stop_words_extra=stops,verbose=False,download_nltk=False) for x in string_list]
    tmp_data.append(tmp_list)
abstracts_pure = tmp_data.copy()
del tmp_data

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

# =============================================================================
# Thesaurus matching
# =============================================================================
print("\nThesaurus matching")

abstracts = kw.thesaurus_matching(abstracts)
abstracts_pure = kw.thesaurus_matching(abstracts_pure)
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


# =============================================================================
# Remove substrings : 
#   be careful with this one! It might remove parts of a string or half of a word
# =============================================================================
thesaurus = pd.read_csv('to_remove.csv')
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

# =============================================================================
# Write to disk
# =============================================================================
corpus_abstract = pd.DataFrame(corpus_abstract,columns=['words'])
corpus_abstract_tr = pd.DataFrame(corpus_abstract_tr,columns=['words'])
corpus_abstract_pure = pd.DataFrame(corpus_abstract_pure,columns=['words'])
corpus_abstract_pure_tr = pd.DataFrame(corpus_abstract_pure_tr,columns=['words'])

corpus_abstract.to_csv(root_dir+subdir+'abstract-title-keys/'+str(year_from)+'-'+str(year_to-1)+' corpus abstract-title-keys',index=False,header=False)
corpus_abstract_tr.to_csv(root_dir+subdir+'abstract-title-keys unwanted-terms-removed/'+str(year_from)+'-'+str(year_to-1)+' corpus abstract-title-keys popular-terms-removed' ,index=False,header=False)
corpus_abstract_pure.to_csv(root_dir+subdir+'abstract-title/'+str(year_from)+'-'+str(year_to-1)+' corpus abstract-title',index=False,header=False)
corpus_abstract_pure_tr.to_csv(root_dir+subdir+'abstract-title unwanted-terms-removed/'+str(year_from)+'-'+str(year_to-1)+' corpus abstract-title popular-terms-removed',index=False,header=False)
