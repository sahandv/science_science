#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 14:56:06 2020

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

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
tqdm.pandas()
nltk.download('wordnet')
nltk.download('punkt')
# =============================================================================
# Read data and Initialize
# =============================================================================
year_from = 0
year_to = 2021

MAKE_SENTENCE_CORPUS = False
MAKE_SENTENCE_CORPUS_ADVANCED_KW = False
MAKE_SENTENCE_CORPUS_ADVANCED = False
MAKE_REGULAR_CORPUS = True
GET_WORD_FREQ_IN_SENTENCE = False
PROCESS_KEYWORDS = False

stops = ['a','an','we','result','however','yet','since','previously','although','propose','proposed','this','...']
nltk.download('stopwords')
stop_words = list(set(stopwords.words("english")))+stops

data_path_rel = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/Corpus/cora-classify/cora/extractions_with_unique_id_labeled.csv'
citations_path = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/Corpus/cora-classify/cora/citations_filtered.csv'
# data_path_rel = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/AI 4k/scopus_4k.csv'
# data_path_rel = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/AI ALL 1900-2019 - reformat'
# data_path_rel = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/AI 300/merged - scopus_v2_relevant wos_v1_relevant - duplicate doi removed - abstract corrected - 05 Aug 2019.csv'
data_full_relevant = pd.read_csv(data_path_rel)
citations = pd.read_csv(citations_path)
# data_full_relevant = data_full_relevant[['dc:title','authkeywords','abstract','year']]
# data_full_relevant.columns = ['TI','DE','AB','PY']
sample = data_full_relevant.sample(4)

root_dir = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/Corpus/cora-classify/cora/'
subdir = 'clean/with citations new/' # no_lemmatization_no_stopwords
gc.collect()

data_full_relevant['PY'] = 2010
data_full_relevant['AB'] = data_full_relevant['Abstract']
data_full_relevant['TI'] = data_full_relevant['Title']
data_full_relevant['DE'] = np.nan
data_full_relevant['ID'] = ''
data_full_relevant['SO'] = ''

# 
data_wrong = data_full_relevant[data_full_relevant['AB'].str.contains("abstract available")].index
data_wrong = list(data_wrong)
data_full_relevant = data_full_relevant.drop(data_wrong,axis=0)
sample = data_full_relevant.sample(4)

# =============================================================================
# filter by citations
# =============================================================================
all_citations = citations['referring_id'].values.tolist()
all_citations.extend(citations['cited_id'].values.tolist())
all_citations = pd.DataFrame(all_citations)
all_citations.columns = ['citation_id']
citations_mask = list(all_citations.groupby('citation_id').groups.keys())

data_full_relevant = data_full_relevant[data_full_relevant['id'].isin(citations_mask)]
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

# Remove special chars and strings from abstracts
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_c(x) if pd.notnull(x) else np.nan).str.lower()
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'et al.') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'eg.') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'ie.') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'vs.') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'ieee') if pd.notnull(x) else np.nan)
data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.find_and_remove_term(x,'fig.','figure') if pd.notnull(x) else np.nan)

# Remove numbers from abstracts to eliminate decimal points and other unnecessary data
# gc.collect()
abstracts = []
for abstract in tqdm(data_with_abstract['AB'].values.tolist()):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", abstract)
    for number in numbers:
        abstract = kw.find_and_remove_term(abstract,number)
    abstracts.append(abstract)
data_with_abstract['AB'] = abstracts.copy()
del  abstracts

alist = pd.DataFrame(data_with_abstract['Author'].values.tolist(),columns=['author'])
alist.to_csv(root_dir+subdir+'corpus authors',index=False) # Save year indices to disk for further use

class_list = pd.DataFrame(data_with_abstract['class1'].values.tolist(),columns=['class1'])
class_list.to_csv(root_dir+subdir+'corpus classes1',index=False) # Save year indices to disk for further use

id_list = pd.DataFrame(data_with_abstract['id'].values.tolist(),columns=['id'])
id_list.to_csv(root_dir+subdir+'corpus idx',index=False) # Save year indices to disk for further use

gc.collect()
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
    
    # del data_with_abstract
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
    sentence_df.to_csv(root_dir+subdir+str(year_from)+'-'+str(year_to-1)+' corpus sentences abstract-title',index=False,header=True)
    
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
    
    file = root_dir+subdir+str(year_from)+'-'+str(year_to-1)+' corpus abstract-title'#'/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/corpus/AI ALL/1900-2019 corpus sentences abstract-title'
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
    tmp_list = [kw.string_pre_processing(x,stemming_method='None',lemmatization='ALL',stop_word_removal=True,stop_words_extra=stops,verbose=False,download_nltk=False) for x in string_list]
    tmp_data.append(tmp_list)
abstracts = tmp_data.copy()
del tmp_data

tmp_data = []
for string_list in tqdm(abstracts_pure, total=len(abstracts_pure)):
    tmp_list = [kw.string_pre_processing(x,stemming_method='None',lemmatization=False,stop_word_removal=False,stop_words_extra=stops,verbose=False,download_nltk=False) for x in string_list]
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
# Remove single letter words
# =============================================================================
tmp = []
for abstract in tqdm(abstracts):
    tmp.append([x for x in abstract if len(x)>1])

abstracts = tmp.copy()
# =============================================================================
# Term to string corpus for co-word analysis
# =============================================================================
print("\nTerm to string corpus")
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
corpus_abstract_pure.shape
corpus_abstract.to_csv(root_dir+subdir+'abstract_title all-lem',index=False,header=False)
# corpus_abstract_tr.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' abstract_title_keys-terms_removed' ,index=False,header=False)
corpus_abstract_pure.to_csv(root_dir+subdir+'abstract_title very pure',index=False,header=False)
# corpus_abstract_pure_tr.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' abstract_title-terms_removed',index=False,header=False)
# corpus_keywords.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' keywords',index=False,header=False)
# corpus_keywords_tr.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' keywords-terms_removed',index=False,header=False)
# corpus_keywords_index.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' keywords_index',index=False,header=False)
# corpus_keywords_index_tr.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' keywords_index-terms_removed',index=False,header=False)