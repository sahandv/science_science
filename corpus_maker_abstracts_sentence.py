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

MAKE_SENTENCE_CORPUS = False
MAKE_REGULAR_CORPUS = True
GET_WORD_FREQ_IN_SENTENCE = False

stops = ['a','an','we','result','however','yet','since','previously','although','propose','proposed','this']
nltk.download('stopwords')
stop_words = list(set(stopwords.words("english")))+stops

data_path_rel = '/home/sahand/GoogleDrive/Data/Relevant Results _ DOI duplication - scopus keywords - document types - 31 july.csv'
#data_path_rel = '/mnt/6016589416586D52/Users/z5204044/Documents/Dataset/Scopus/AI/ALL/processed/AI ALL 1900-2019 - reformat'
data_full_relevant = pd.read_csv(data_path_rel)

root_dir = '/home/sahand/GoogleDrive/Data/Corpus/'
subdir = 'copyr_lemmatized_stopword_removed_thesaurus/' # no_lemmatization_no_stopwords


# =============================================================================
# Initial Pre-Processing : 
#   Following tags requires WoS format. Change them otherwise.
# =============================================================================
data_filtered = data_full_relevant.copy()
data_filtered = data_filtered[pd.notnull(data_filtered['PY'])]

data_filtered = data_filtered[data_filtered['PY'].astype('int')>year_from-1]
data_filtered = data_filtered[data_filtered['PY'].astype('int')<year_to]

# Remove columns without keywords/abstract list 
data_with_keywords = data_filtered[pd.notnull(data_filtered['DE'])]
data_with_abstract = data_filtered[pd.notnull(data_filtered['AB'])]

data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_c(x) if pd.notnull(x) else np.nan)

year_list = pd.DataFrame(data_with_abstract['PY'].values.tolist(),columns=['year'])
year_list.to_csv(root_dir+subdir+str(year_from)+'-'+str(year_to-1)+' years',index=False) # Save year indices to disk for further use
# =============================================================================
# Sentence maker
# =============================================================================
if MAKE_SENTENCE_CORPUS is True:
    thesaurus = pd.read_csv('data/thesaurus/thesaurus_for_ai_keyword_with_().csv')
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

if MAKE_REGULAR_CORPUS is False:
    sys.exit('Did not continue to create normal corpus. If you want a corpus, set it to True at init section.')

# =============================================================================
#   Get word frequency in sentence corpus -- OPTIONAL
# =============================================================================
if GET_WORD_FREQ_IN_SENTENCE is True:
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    
    file = root_dir+subdir+str(year_from)+'-'+str(year_to-1)+' corpus sentences abstract-title'#'/home/sahand/GoogleDrive/Data/corpus/AI ALL/1900-2019 corpus sentences abstract-title'
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
print("\nString pre processing for abstracts")
for string_list in tqdm(abstracts, total=len(abstracts)):
    tmp_list = [kw.string_pre_processing(x,stemming_method='None',lemmatization=True,stop_word_removal=True,stop_words_extra=stops,verbose=False,download_nltk=False) for x in string_list]
    tmp_data.append(tmp_list)
abstracts = tmp_data.copy()
del tmp_data

tmp_data = []
for string_list in tqdm(abstracts_pure, total=len(abstracts_pure)):
    tmp_list = [kw.string_pre_processing(x,stemming_method='None',lemmatization=True,stop_word_removal=True,stop_words_extra=stops,verbose=False,download_nltk=False) for x in string_list]
    tmp_data.append(tmp_list)
abstracts_pure = tmp_data.copy()
del tmp_data

print("\nString pre processing for keywords")
tmp_data = []
for string_list in tqdm(keywords, total=len(keywords)):
    tmp_list = []
    for string in string_list:
        tmp_sub_list = string.split()
        tmp_list.append(' '.join([kw.string_pre_processing(x,stemming_method='None',lemmatization=True,stop_word_removal=True,stop_words_extra=stops,verbose=False,download_nltk=False) for x in tmp_sub_list]))
    tmp_data.append(tmp_list)
keywords = tmp_data.copy()
del tmp_data

tmp_data = []
for string_list in tqdm(keywords_index, total=len(keywords_index)):
    tmp_list = []
    for string in string_list:
        tmp_sub_list = string.split()
        tmp_list.append(' '.join([kw.string_pre_processing(x,stemming_method='None',lemmatization=True,stop_word_removal=True,stop_words_extra=stops,verbose=False,download_nltk=False) for x in tmp_sub_list]))
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

abstracts = kw.thesaurus_matching(abstracts)
abstracts_pure = kw.thesaurus_matching(abstracts_pure)
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

corpus_abstract.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' abstract_title_keys',index=False,header=False)
corpus_abstract_tr.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' abstract_title_keys-terms_removed' ,index=False,header=False)
corpus_abstract_pure.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' abstract_title',index=False,header=False)
corpus_abstract_pure_tr.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' abstract_title-terms_removed',index=False,header=False)
corpus_keywords.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' keywords',index=False,header=False)
corpus_keywords_tr.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' keywords-terms_removed',index=False,header=False)
corpus_keywords_index.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' keywords_index',index=False,header=False)
corpus_keywords_index_tr.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' keywords_index-terms_removed',index=False,header=False)