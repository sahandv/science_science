#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 14:56:06 2020

@author: github.com/sahandv
"""
import sys
import gc
import json
import string

from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from sciosci.assets import text_assets as kw
from sciosci.assets import keyword_dictionaries as kd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import strip_multiple_whitespaces
tqdm.pandas()
nltk.download('wordnet')
nltk.download('punkt')
# =============================================================================
# Read data and Initialize
# =============================================================================
year_from = 1960
year_to = 2021

abstract_length_min = 300 #character lower bound
abstract_length_min_w = 60 # word lower bound

MAKE_SENTENCE_CORPUS = False
MAKE_SENTENCE_CORPUS_ADVANCED_KW = False
MAKE_SENTENCE_CORPUS_ADVANCED = False
MAKE_REGULAR_CORPUS = True
GET_WORD_FREQ_IN_SENTENCE = False
PROCESS_KEYWORDS = False

stops = ['a','an','we','result','however','yet','since','previously','although','propose','proposed','this','...']
nltk.download('stopwords')
stop_words = list(set(stopwords.words("english")))+stops

# data_path_rel = '/home/sahand/GoogleDrive/Data/Corpus/Dimensions All/clean/data with abstract'

# data_path_rel = '/home/sahand/GoogleDrive/Data/Corpus/Taxonomy/AI kw merged'
# data_path_rel = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/AI 4k/scopus_4k.csv'
# data_path_rel = '/home/sahand/GoogleDrive/Data/Corpus/Dimensions AI unlimited citations/clean/publication idx'
# data_path_rel = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/Corpus/AI 300/merged - scopus_v2_relevant wos_v1_relevant - duplicate doi removed - abstract corrected - 05 Aug 2019.csv'

with open('/home/sahand/GoogleDrive/Data/Corpus/Dimensions All/1961-2020 dimensions AI articles-proceedings.json') as f:
    data = json.load(f)
data_full_relevant = pd.DataFrame(data)
del data
gc.collect()

# data_full_relevant = pd.read_csv(data_path_rel)
# data_full_relevant = pd.read_csv(data_path_rel,names=['abstract'])


# data_full_relevant = data_full_relevant[['dc:title','authkeywords','abstract','year']]
# data_full_relevant.columns = ['TI','DE','AB','PY']

root_dir = '/home/sahand/GoogleDrive/Data/Corpus/Dimensions All/'
subdir = 'clean/' # no_lemmatization_no_stopwords
gc.collect()

data_full_relevant['PY'] = data_full_relevant['year']
data_full_relevant['AB'] = data_full_relevant['abstract'].str.lower()
data_full_relevant['TI'] = data_full_relevant['title'].str.lower()
data_full_relevant['DE'] = np.nan
data_full_relevant['ID'] = ''
data_full_relevant['SO'] = data_full_relevant['journal']

# data_filtered = pd.read_csv('/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/Corpus/Dimensions/abstract_title deflemm',names=['AB'])
# data_wrong = data_full_relevant[data_full_relevant['AB'].str.contains("abstract available")].index
# data_wrong = list(data_wrong)
# data_full_relevant = data_full_relevant.drop(data_wrong,axis=0)
sample = data_full_relevant.sample(400)
# tmp_concept = data_full_relevant['concepts_scores'][287041]
# tmp_abstract = data_full_relevant['abstract'][287041]
# tmp_terms = data_full_relevant['terms'][287041]
# tmp_for = data_full_relevant['category_for'][106041]

# path = '/home/sahand/GoogleDrive/Data/'
# pub_idx = pd.read_csv(path+'Corpus/Dimensions AI unlimited citations/clean/publication idx')
# data_full_relevant = data_full_relevant[data_full_relevant['id'].isin(pub_idx['id'].values.tolist())]

# =============================================================================
# Initial Pre-Processing : 
#   Following tags requires WoS format. Change them otherwise.
# =============================================================================
data_filtered = data_full_relevant[['category_for','PY','SO','AB','TI','reference_ids','authors','concepts_scores','id']]
sample = data_filtered.sample(10000)


# data_filtered = data_full_relevant.copy()
data_filtered = data_filtered[pd.notnull(data_filtered['PY'])]

# data_filtered = data_filtered[data_filtered['PY'].astype('int')>year_from-1]
# data_filtered = data_filtered[data_filtered['PY'].astype('int')<year_to]

# Remove columns without keywords/abstract list 
data_with_abstract = data_filtered[pd.notnull(data_filtered['AB'])]
data_with_abstract = data_with_abstract[pd.notnull(data_with_abstract['AB'])]
data_with_abstract = data_with_abstract[pd.notnull(data_with_abstract['id'])]
data_with_abstract = data_with_abstract[pd.notnull(data_with_abstract['TI'])]
data_with_abstract = data_with_abstract[pd.notnull(data_with_abstract['id'])]
data_with_abstract = data_with_abstract[pd.notnull(data_with_abstract['authors'])]
data_with_abstract = data_with_abstract[pd.notnull(data_with_abstract['PY'])]
data_with_abstract = data_with_abstract[pd.notnull(data_with_abstract['reference_ids'])]
data_with_abstract = data_with_abstract[pd.notnull(data_with_abstract['concepts_scores'])]

# =============================================================================
# remove repeating ids
# =============================================================================
data_with_abstract = data_with_abstract.drop_duplicates(subset=['id']).reset_index(drop=True)

# del data_full_relevant
del data_filtered

# =============================================================================
# Further cleaning - optional
# =============================================================================
# Remove special chars and strings from abstracts
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

# num2words = {' 1 ': ' one ', '.1-': '.one-', ' 1-': ' one-', '-1 ': '-one ', '-1-': '-one-', 
#              ' 2 ': ' two ',  '.2-': '.two-', ' 2-': ' two-', '-2 ': '-two ', '-2-': '-two-', 
#              ' 3 ': ' three ',  '.3-': '.three-', ' 3-': ' three-', '-3 ': '-three ', '-3-': '-three-', 
#              ' 4 ': ' four ',  '.4-': '.four-', ' 4-': ' four-', '-4 ': '-four ', '-4-': '-four-', 
#              ' 5 ': ' five ',  '.5-': '.five-', ' 5-': ' five-', '-5 ': '-five ', '-5-': '-five-', 
#              ' 6 ': ' six ',  '.6-': '.six-', ' 6-': ' six-', '-6 ': '-six ', '-6-': '-six-', 
#              ' 7 ': ' seven ',  '.7-': '.seven-', ' 7-': ' seven-', '-7 ': '-seven ', '-7-': '-seven-', 
#              ' 8 ': ' eight ',  '.8-': '.eight-', ' 8-': ' eight-', '-8 ': '-eight ', '-8-': '-eight-', 
#              ' 9 ': ' nine ', '.9-': '.nine-', ' 9-': ' nine-', '-9 ': '-nine ', '-9-': '-nine-'}

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


# data_with_abstract['AB'] = data_with_abstract['AB'].progress_apply(lambda x: kw.multiple_replace(x,num2words) if pd.notnull(x) else np.nan)
# data_with_abstract['TI'] = data_with_abstract['TI'].progress_apply(lambda x: kw.multiple_replace(x,num2words) if pd.notnull(x) else np.nan)

# =============================================================================
# 
# =============================================================================
# Remove numbers from abstracts to eliminate decimal points and other unnecessary data
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
gc.collect()

data_with_abstract = data_with_abstract[pd.notnull(data_with_abstract['AB'])]

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

# data_with_abstract[['id']].to_csv(root_dir+subdir+'mask',index=False)

# =============================================================================
# Language check
# =============================================================================
def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
    
# Fix titles
other_langs = []
for i,row in tqdm(data_with_abstract.iterrows(),total=data_with_abstract.shape[0]):
    if not isEnglish(row['TI']):
        other_langs.append(row['id'])

corrected_text = data_with_abstract[data_with_abstract['id'].isin(other_langs)]['TI'].progress_apply(lambda text: ''.join(x for x in text if x in string.printable))
data_with_abstract.loc[data_with_abstract['id'].isin(other_langs),'TI'] = corrected_text

# Check Abstracts
other_langs = []
for i,row in tqdm(data_with_abstract.iterrows(),total=data_with_abstract.shape[0]):
    if not isEnglish(row['AB']):
        other_langs.append(row['id'])

corrected_text = data_with_abstract[data_with_abstract['id'].isin(other_langs)]['AB'].progress_apply(lambda text: ''.join(x for x in text if x in string.printable))
data_with_abstract.loc[data_with_abstract['id'].isin(other_langs),'AB'] = corrected_text
# bad_data = data_with_abstract.loc[data_with_abstract['id'].isin(other_langs),'AB']
# kw.find_and_remove_term(text,'\n',' ')
# ''.join(x for x in text if x in string.printable)
# =============================================================================
# concept preparation from json
# =============================================================================
def extract_concepts(concepts):
    try:
        concepts_list = []
        for concept in concepts:
            try:
                concepts_list.append(concept['concept']+':::'+str(concept['relevance']))
            except:
                pass
                # print('oops! concept was empty...')
        return ';;;'.join(concepts_list)
    except:
        print('oops!')
        return np.nan
data_with_abstract['DE'] = data_with_abstract['concepts_scores'].progress_apply(lambda x: extract_concepts(x))
# =============================================================================
# Ref preparation from json
# =============================================================================
def author_to_str(authors):
    try:
        return ';;;'.join(authors)
    except:
        return np.nan
    
data_with_abstract['reference_ids_str'] = data_with_abstract['reference_ids'].progress_apply(lambda x: author_to_str(x))
data_with_abstract['reference_ids_str'][100]

# =============================================================================
# Journal preparation from json
# =============================================================================
data_with_abstract['SO'] = data_with_abstract['SO'].progress_apply(lambda x: {'id':np.nan,'title':np.nan} if pd.isnull(x) else x)
data_with_abstract['journal_name'] = [x['title'] for x in tqdm(data_with_abstract['SO'].values.tolist())]
data_with_abstract['journal_id'] = [x['id'] for x in tqdm(data_with_abstract['SO'].values.tolist())]
# =============================================================================
# category FOR preparation from json
# =============================================================================
def extract_cats(cats):
    try:
        cat_list = []
        for cat in cats:
            try:
                cat_list.append(cat['id']+':::'+str(cat['name']))
            except:
                pass
                # print('oops! concept was empty...')
        return ';;;'.join(cat_list)
    except:
        print('oops!')
        return np.nan
data_with_abstract['FOR'] = data_with_abstract['category_for'].progress_apply(lambda x: extract_cats(x))

# =============================================================================
# Author preparation from json
# =============================================================================
def extract_authors(authors):
    researcher_ids = []
    try:
        for author in authors:
            try:
                if author['researcher_id']!='':
                    researcher_ids.append(author['researcher_id'])
            except:
                pass
        return ';;;'.join(researcher_ids)
    except:
        return np.nan
    
data_with_abstract['reseearcher_ids'] = data_with_abstract['authors'].progress_apply(lambda x: extract_authors(x))

# auhtor = data_with_abstract['authors'][176279]

data_with_abstract = data_with_abstract.drop(['SO','category_for','reference_ids','authors','concepts_scores'],axis=1)
sample = data_with_abstract.sample(100)

# =============================================================================
# Author preparation from csv
# =============================================================================
authors_j = []
errors = []
for i,row in tqdm(data_with_abstract.iterrows(),total=data_with_abstract.shape[0]):
    row_j = []
    if pd.notna(data_with_abstract['authors'][i]):
    # for line in data_with_abstract['authors'][i].replace('"','').replace('"','').replace('[','["').replace(']','"]').replace("'",'"').replace('"[','[').replace(']"',']').replace('None','""').replace('[""','["').replace('""]','"]').replace('["{','[{').replace('}"]','}]')[2:-2].split('}, {'):
        try:
            row = kw.remove_substring_content(data_with_abstract['authors'][i].replace("{'",'{"').replace("'}",'"}').replace("['",'["').replace("']",'"]').replace("':",'":').replace("' :",'" :').replace(":'",':"').replace(": '",': "').replace(",'",',"').replace(", '",', "').replace("',",'",').replace("' ,",'" ,').replace('None','""').replace('True','"True"').replace('False','"False"').replace('"[','[').replace(']"',']')[2:-2],a='[',b=']',replace='""')
            for line in row.split('}, {'):
                row_j.append(json.loads('{'+line+'}'))
            authors_j.append(row_j)
        except:
            authors_j.append(np.nan)
            errors.append(i)
    else:
        authors_j.append(np.nan)

data_for_authors = []
for i,pub in tqdm(enumerate(authors_j),total=len(authors_j)):
    pub_id = data_with_abstract['id'][i]
    try:
        for auth in authors_j[i]:
            data_for_authors.append([pub_id,auth['first_name'],auth['last_name'],auth['orcid'],auth['current_organization_id'],auth['researcher_id']])
    except:
        pass

data_for_authors = pd.DataFrame(data_for_authors,columns=['pub_id','first_name','last_name','orcid','current_organization_id','researcher_id']) 
data_for_authors = data_for_authors[data_for_authors['researcher_id']!='']
data_for_authors.to_csv(root_dir+subdir+'authors with research_id',index=False) # Save year indices to disk for further use

pubs_with_r_id = list(data_for_authors.groupby('pub_id').groups.keys())
# =============================================================================
# filter data by id from previously created filters-- optional
# =============================================================================
id_filter = pd.read_csv(root_dir+subdir+'publication idx')['id'].values.tolist()

data_with_abstract = data_with_abstract[data_with_abstract['id'].isin(id_filter)]
data_with_abstract = data_with_abstract[data_with_abstract['id'].isin(pubs_with_r_id)]
data_with_abstract = data_with_abstract.reset_index(drop=True)



# filtered_abstracts_lem = pd.read_csv(root_dir+subdir+'abstract_title deflemm',names=['abstract'])
# filtered_abstracts_lem['id'] = data_with_abstract['id']
# filtered_abstracts_lem = filtered_abstracts_lem[filtered_abstracts_lem['id'].isin(id_filter)]
# filtered_abstracts_lem.to_csv(root_dir+subdir+'abstract_title deflemm',index=False)

# filtered_abstracts_pure = pd.read_csv(root_dir+subdir+'abstract_title pure',names=['abstract'])
# filtered_abstracts_pure['id'] = data_with_abstract['id']
# filtered_abstracts_pure = filtered_abstracts_pure[filtered_abstracts_pure['id'].isin(id_filter)]
# filtered_abstracts_pure.to_csv(root_dir+subdir+'abstract_title pure',index=False)

# filtered_abstracts = pd.read_csv(root_dir+subdir+'abstract_title pure US with id',names=['abstract','id'])
# filtered_abstracts = filtered_abstracts[filtered_abstracts['id'].isin(id_filter)]
# filtered_abstracts.to_csv(root_dir+subdir+'abstract_title pure US with id',index=False)

# filtered_cats_masked = pd.read_csv(root_dir+subdir+'categories_masked_clean')
# filtered_cats_masked['id'] = data_with_abstract['id']
# filtered_cats_masked = filtered_cats_masked[filtered_cats_masked['id'].isin(id_filter)]
# filtered_cats_masked.to_csv(root_dir+subdir+'categories_masked_clean',index=False)

# filtered_cats_processed = pd.read_csv(root_dir+subdir+'categories_processed')
# filtered_cats_processed['id'] = data_with_abstract['id']
# filtered_cats_processed = filtered_cats_processed[filtered_cats_processed['id'].isin(id_filter)]
# filtered_cats_processed.to_csv(root_dir+subdir+'categories_processed',index=False)

# filtered_refs = pd.read_csv(root_dir+subdir+'corpus references')
# filtered_refs['id'] = data_with_abstract['id']
# filtered_refs = filtered_refs[filtered_refs['id'].isin(id_filter)]
# filtered_refs.to_csv(root_dir+subdir+'corpus references',index=False)

# filtered_so = pd.read_csv(root_dir+subdir+'corpus sources')
# filtered_so['id'] = data_with_abstract['id']
# filtered_so = filtered_so[filtered_so['id'].isin(id_filter)]
# filtered_so.to_csv(root_dir+subdir+'corpus sources',index=False)

# filtered_py = pd.read_csv(root_dir+subdir+'corpus years')
# filtered_py['id'] = data_with_abstract['id']
# filtered_py = filtered_py[filtered_py['id'].isin(id_filter)]
# filtered_py.to_csv(root_dir+subdir+'corpus years',index=False)

# =============================================================================
# Journal/Book category preparation and check -- optional
# =============================================================================
types = []
sources = []
errors = []
journal_title = []
journal_id = []
conference_title = []
for i,row in tqdm(data_with_abstract.iterrows(),total=data_with_abstract.shape[0]):
    try:
        output = json.loads(data_with_abstract['journal'][i].replace("{'",'{"').replace("'}",'"}').replace("':",'":').replace("' :",'" :').replace(":'",':"').replace(": '",': "').replace(",'",',"').replace(", '",', "').replace("',",'",').replace("' ,",'" ,').replace('None','""').replace('True','"True"').replace('False','"False"'))
        types.append('j')
        journal_title.append(output['title'])
        journal_id.append(output['id'])
        conference_title.append(np.nan)
    except:
        try:
            output = data_with_abstract['proceedings_title'][i].replace("{'",'{"').replace("'}",'"}').replace("':",'":').replace("' :",'" :').replace(":'",':"').replace(": '",': "').replace(",'",',"').replace(", '",', "').replace("',",'",').replace("' ,",'" ,').replace('None','""').replace('True','"True"').replace('False','"False"')
            types.append('c')
            journal_title.append(np.nan)
            journal_id.append(np.nan)
            conference_title.append(output)
        except:
            output = np.nan
            errors.append([i,row])
            types.append('u')
            journal_title.append(np.nan)
            journal_id.append(np.nan)
            conference_title.append(np.nan)
            
    sources.append(output)

data_with_abstract['journal_title'] = journal_title
data_with_abstract['journal_id'] = journal_id
data_with_abstract['conference_title'] = conference_title

journal_groups = data_with_abstract.groupby('journal_title').groups
journal_groups_keys = journal_groups.keys()

cat_check = data_with_abstract.iloc[list(journal_groups['Технология и конструирование в электронной аппаратуре'])]

# =============================================================================
# keyword preparation
# =============================================================================
data_with_abstract['DE-n'] = data_with_abstract['DE'].progress_apply(lambda x: x.split(';;;') if pd.notnull(x) else np.nan)
data_with_abstract = data_with_abstract[data_with_abstract['DE-n'].notna()]
data_with_abstract = data_with_abstract.reset_index(drop=True)
data_with_abstract['DE-n'] = data_with_abstract['DE-n'].progress_apply(lambda x: [a.split(':::')[0] for a in x])
# data_with_abstract['DE-terms'] =  data_with_abstract['DE-n'].progress_apply(lambda x: ','.join([a[0] for a in x]))
data_with_abstract['DE-n'] = data_with_abstract['DE-n'].progress_apply(lambda x: [strip_multiple_whitespaces(a).strip().lower() for a in x])
data_with_abstract['DE-n'] = data_with_abstract['DE-n'].progress_apply(lambda x: [kw.string_pre_processing(a,stemming_method='None',lemmatization='DEF',stop_word_removal=True,stop_words_extra=stops,verbose=False,download_nltk=False) for a in x])
data_with_abstract['DE-n'] = data_with_abstract['DE-n'].progress_apply(lambda x: [kw.replace_british_american(a,kd.gb2us) for a in x]) # Optional step
data_with_abstract['DE-n'] = data_with_abstract['DE-n'].progress_apply(lambda x: ';;;'.join(x)) # Optional step



# =============================================================================
# Simple pre-process (method b) -- optional -- preferred for kw extraction
# =============================================================================
data_with_abstract['TI_AB_b'] = data_with_abstract.TI.map(str) + ". " + data_with_abstract.AB
abstracts = [re.sub('[^A-Za-z0-9 .?,!()]','',ab) for ab in data_with_abstract['TI_AB_b']]
abstracts = [strip_multiple_whitespaces(ab).strip().lower() for ab in tqdm(abstracts)]

thesaurus = [
    [' 1)',', '],
    [' 2)',', '],
    [' 3)',', '],
    [' 4)',', '],
    [' 5)',', '],
    [' 6)',', '],
    [' 7)',', '],
    [' 8)',', '],
    [' 9)',', '],
    [' a)',', '],
    [' b)',', '],
    [' c)',', '],
    [' d) ',', '],
    [' e)',', '],
    [' f)',', '],
    [' g)',', '],
    [' h)',', '],
    [' a. ',', '],
    [' b. ',', '],
    [' c. ',', '],
    [' d. ',', '],
    [' e. ',', '],
    [' f. ',', '],
    [' g. ',', '],
    [' h. ',', '],
    [' i)',', '],
    [' ii)',', '],
    [' iii)',', '],
    [' iv)',', '],
    [' v)',', '],
    [' vi)',', '],
    [' vii)',', '],
    [' viii)',', '],
    [' ix)',', '],
    [' x)',', '],
    [' xi)',', '],
    [' xii)',', '],
    [' i. ',', '],
    [' ii. ',', '],
    [' iii. ',', '],
    [' iv. ',', '],
    [' v. ',', '],
    [' vi. ',', '],
    [' vii. ',', '],
    [' viii. ',', '],
    [' ix. ',', '],
    [' x. ',', '],
    [' xi. ',', '],
    [' xii. ',', '],
    [' i.e.',', '],
    [' ie.',', '],
    [' eg.',', '],
    [' e.g.',', ']
    ]

tmp = []
for paragraph in tqdm(abstracts):
    paragraph = kw.filter_string(paragraph,thesaurus)
    tmp.append(paragraph)
abstracts = tmp

abstracts = [strip_multiple_whitespaces(ab).strip().lower() for ab in tqdm(abstracts)]

thesaurus = [
    [',,',',']
    ]

tmp = []
for paragraph in tqdm(abstracts):
    paragraph = kw.filter_string(paragraph,thesaurus)
    tmp.append(paragraph)
abstracts = tmp
# data_with_abstract['TI_AB_b'] = abstracts
# abstracts = data_with_abstract['TI_AB_b'].values.tolist()
abstracts = [kw.string_pre_processing(x,stemming_method='None',lemmatization='DEF',stop_word_removal=True,stop_words_extra=stops,verbose=False,download_nltk=False) for x in tqdm(abstracts)]

data_with_abstract['TI_AB_b'] = abstracts
corpus_abstract_pure_df = pd.DataFrame(abstracts,columns=['abstract'])
corpus_abstract_pure_df['id'] = data_with_abstract['id']
corpus_abstract_pure_df.to_csv(root_dir+subdir+'abstract_title method_b_3',index=False)
sample = corpus_abstract_pure_df.sample(100)
# =============================================================================
# Simple pre-process (method a) -- optional -- preferred for deep learning of documents
# =============================================================================
data_with_abstract['TI_AB'] = data_with_abstract.TI.map(str) + ". " + data_with_abstract.AB
abstracts = [re.sub('[^A-Za-z0-9 .?,!()]','',ab) for ab in data_with_abstract['TI_AB']]
abstracts = [strip_multiple_whitespaces(ab).strip().lower() for ab in abstracts]
tmp = []
errors = []
for i,abstract in tqdm(enumerate(abstracts),total=len(abstracts)):
    try:
        tmp.append(kw.replace_british_american(kw.replace_british_american(strip_multiple_whitespaces(abstract),kd.gb2us),kd.gb2us))
    except:
        tmp.append('')
        errors.append(i)
corpus_abstract_pure = tmp


thesaurus = [
    [' 1)',', '],
    [' 2)',', '],
    [' 3)',', '],
    [' 4)',', '],
    [' 5)',', '],
    [' 6)',', '],
    [' 7)',', '],
    [' 8)',', '],
    [' 9)',', '],
    [' a)',', '],
    [' b)',', '],
    [' c)',', '],
    [' d) ',', '],
    [' e)',', '],
    [' f)',', '],
    [' g)',', '],
    [' h)',', '],
    [' a. ',', '],
    [' b. ',', '],
    [' c. ',', '],
    [' d. ',', '],
    [' e. ',', '],
    [' f. ',', '],
    [' g. ',', '],
    [' h. ',', '],
    [' i)',', '],
    [' ii)',', '],
    [' iii)',', '],
    [' iv)',', '],
    [' v)',', '],
    [' vi)',', '],
    [' vii)',', '],
    [' viii)',', '],
    [' ix)',', '],
    [' x)',', '],
    [' xi)',', '],
    [' xii)',', '],
    [' i. ',', '],
    [' ii. ',', '],
    [' iii. ',', '],
    [' iv. ',', '],
    [' v. ',', '],
    [' vi. ',', '],
    [' vii. ',', '],
    [' viii. ',', '],
    [' ix. ',', '],
    [' x. ',', '],
    [' xi. ',', '],
    [' xii. ',', '],
    [' i.e.',', '],
    [' ie.',', '],
    [' eg.',', '],
    [' e.g.',', ']
    ]

tmp = []
for paragraph in tqdm(corpus_abstract_pure):
    paragraph = kw.filter_string(paragraph,thesaurus)
    tmp.append(paragraph)
corpus_abstract_pure_final = tmp

corpus_abstract_pure_final = [strip_multiple_whitespaces(ab).strip().lower() for ab in corpus_abstract_pure_final]

thesaurus = [
    [',,',',']
    ]

tmp = []
for paragraph in tqdm(corpus_abstract_pure_final):
    paragraph = kw.filter_string(paragraph,thesaurus)
    tmp.append(paragraph)
corpus_abstract_pure_final = tmp
ch = '.'
corpus_abstract_pure_final = [x.lstrip(ch).lstrip(ch).lstrip(ch).lstrip(ch) for x in corpus_abstract_pure_final]
corpus_abstract_pure_final = [x.lstrip(ch).lstrip(ch).lstrip(ch).lstrip(ch) for x in corpus_abstract_pure_final]


data_with_abstract['TI_AB'] = corpus_abstract_pure_final
corpus_abstract_pure_df = pd.DataFrame(corpus_abstract_pure_final,columns=['abstract'])
corpus_abstract_pure_df['id'] = data_with_abstract['id']

corpus_abstract_pure_df.to_csv(root_dir+subdir+'abstract_title method_a',index=False)

# =============================================================================
# Save to disk
# =============================================================================
data_with_abstract.to_csv(root_dir+subdir+'data with abstract',index=False)

references = pd.DataFrame(data_with_abstract['reference_ids'].values.tolist(),columns=['reference_ids'])
references.to_csv(root_dir+subdir+'corpus references',index=False) # Save year indices to disk for further use

source_list = pd.DataFrame(data_with_abstract['SO'].values.tolist(),columns=['source'])
source_list.to_csv(root_dir+subdir+'corpus sources',index=False) # Save year indices to disk for further use

class_list = pd.DataFrame(data_with_abstract['category_for'].values.tolist(),columns=['category_for'])
class_list.to_csv(root_dir+subdir+'corpus category_for',index=False) # Save year indices to disk for further use

id_list = pd.DataFrame(data_with_abstract['id'].values.tolist(),columns=['id'])
id_list.to_csv(root_dir+subdir+'publication idx',index=False) # Save year indices to disk for further use

id_list = pd.DataFrame(list(data_with_abstract.index),columns=['index'])
id_list.to_csv(root_dir+subdir+'corpus idx',index=False) # Save year indices to disk for further use

year_list = pd.DataFrame(data_with_abstract['PY'].values.tolist(),columns=['year'])
year_list.to_csv(root_dir+subdir+'corpus years',index=False) # Save year indices to disk for further use
gc.collect()

year_list.plot.hist(bins=60, alpha=0.5,figsize=(15,6))
year_list.shape

sample = data_with_abstract.sample(5)

# =============================================================================
# Clean bad data based on abstracts - round 2
# =============================================================================
data_with_abstract['TI_AB'] = corpus_abstract_pure_final

long_abstracts = []
lens = []
word_len = []
for i,row in tqdm(data_with_abstract.iterrows(),total=data_with_abstract.shape[0]):
    leng = len(row['TI_AB'])
    w_leng = len(row['TI_AB'].split())
    word_len.append([len(ab) for ab in row['TI_AB'].split()])
    lens.append(leng)
    if leng > abstract_length_min and w_leng > abstract_length_min_w:
        long_abstracts.append(row['id'])

word_len_f = [j for sub in word_len for j in sub]
median_word_len = np.median(word_len_f)
mean_word_len = np.mean(word_len_f)

max_paragraph_len = int(np.percentile(lens, percentile)) # take Nth percentile as the sentence length threshold
data_with_abstract = data_with_abstract[data_with_abstract['id'].isin(long_abstracts)]
data_with_abstract = data_with_abstract[pd.notnull(data_with_abstract['TI_AB'])]
data_with_abstract = data_with_abstract[pd.notnull(data_with_abstract['id'])]
data_with_abstract = data_with_abstract.drop(['id_n'],axis=1)

data_with_abstract = data_with_abstract.reset_index(drop=True)

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
    
    sentence_corpus.to_csv(root_dir+subdir+' corpus sentences abstract-title',index=False,header=True)

gc.collect()
# =============================================================================
# Sentence maker -- Advanced -- 
# =============================================================================
if MAKE_SENTENCE_CORPUS_ADVANCED is True:    
    data_with_abstract['TI_AB'] = data_with_abstract.TI.map(str) + ". " + data_with_abstract.AB
    data_fresh = data_with_abstract[['TI_AB_b','PY']].copy()
    data_fresh['TI_AB_b'] = data_fresh['TI_AB_b'].str.lower()
    
    # del data_with_abstract
    gc.collect()
    
    data_tmp = data_fresh[1:10]
    data_fresh[-2:-1]

    print("\nSentence extraction")
    sentences = []
    years = []
    indices = []
    for index,row in tqdm(data_fresh.iterrows(),total=data_fresh.shape[0]):
        abstract_str = row['TI_AB_b']
        year = row['PY']
        abstract_sentences = re.split('\. |\? |\\n|\!',abstract_str)
        abstract_sentences = [x for x in abstract_sentences if x!='']
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
    sentence_df.to_csv(root_dir+subdir+'corpus sentences abstract-title-2',index=False,header=True)
    
# =============================================================================
# Keyword Extractor
# =============================================================================
if MAKE_SENTENCE_CORPUS_ADVANCED_KW is True:    
    data_with_abstract['TI_AB'] = data_with_abstract.AB
    data_fresh = data_with_abstract[['TI_AB','PY']].copy()
    data_fresh['TI_AB'] = data_fresh['TI_AB'].str.lower()
    # data_fresh['TI_AB'] = corpus_abstract_pure_final
    
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
    sentence_df.to_csv(root_dir+subdir+' corpus sentences abstract-title',index=False,header=True)
    

if MAKE_REGULAR_CORPUS is False:
    sys.exit('Did not continue to create normal corpus. If you want a corpus, set it to True at init section.')
# =============================================================================
#   Get word frequency in sentence corpus -- OPTIONAL
# =============================================================================
if GET_WORD_FREQ_IN_SENTENCE is True:
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    
    file = root_dir+subdir+' corpus abstract-title'#'/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/corpus/AI ALL/1900-2019 corpus sentences abstract-title'
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
print('Tokenize (Author Keywords and Abstracts+Titles)')
abstracts = []
keywords = []
keywords_index = []
abstracts_pure = []

for index,paper in tqdm(data_with_abstract.iterrows(),total=data_with_abstract.shape[0]):
    keywords_str = paper['DE']
    keywords_index_str = paper['ID']
    abstract_str = kw.replace_british_american(strip_multiple_whitespaces(paper['AB']),kd.gb2us)
    title_str = kw.replace_british_american(strip_multiple_whitespaces(paper['TI']),kd.gb2us)
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
# data_with_abstract['AB_split'] = abstracts_pure 
# data_with_abstract['AB_KW_split'] = abstracts
del data_with_abstract
gc.collect()

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
    tmp_list = [kw.string_pre_processing(x,stemming_method='None',lemmatization='ALL',stop_word_removal=False,stop_words_extra=stops,verbose=False,download_nltk=False) for x in string_list]
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

corpus_abstract_pure = [x.replace(' .','.') for x in corpus_abstract_pure]
corpus_abstract_pure = [x.replace(' ,',',') for x in corpus_abstract_pure]
corpus_abstract_pure = [x.replace(' ;',';') for x in corpus_abstract_pure]


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
# to US english
# =============================================================================
tmp = []
errors = []
for i,abstract in tqdm(enumerate(corpus_abstract_pure),total=len(corpus_abstract_pure)):
    try:
        tmp.append(kw.replace_british_american(strip_multiple_whitespaces(kw.replace_british_american(strip_multiple_whitespaces(abstract),kd.gb2us)),kd.gb2us))
    except:
        tmp.append('')
        errors.append(i)
corpus_abstract_pure = tmp

tmp = []
for abstract in tqdm(corpus_abstract):
    try:
        tmp.append(kw.replace_british_american(strip_multiple_whitespaces(kw.replace_british_american(strip_multiple_whitespaces(abstract),kd.gb2us)),kd.gb2us))
    except:
        tmp.append('')
corpus_abstract = tmp
    
# =============================================================================
# Write to disk
# =============================================================================
corpus_abstract = pd.DataFrame(corpus_abstract,columns=['words'])
# corpus_abstract_tr = pd.DataFrame(corpus_abstract_tr,columns=['words'])
corpus_abstract_pure = pd.DataFrame(corpus_abstract_pure,columns=['words'])
# corpus_abstract_pure_tr = pd.DataFrame(corpus_abstract_pure_tr,columns=['words'])
# corpus_keywords = pd.DataFrame(corpus_keywords,columns=['words'])
# corpus_keywords_tr = pd.DataFrame(corpus_keywords_tr,columns=['words'])
# corpus_keywords_index = pd.DataFrame(corpus_keywords_index,columns=['words'])
# corpus_keywords_index_tr = pd.DataFrame(corpus_keywords_index_tr,columns=['words'])

corpus_abstract.to_csv(root_dir+subdir+'abstract_title deflemm',index=False,header=False)
# corpus_abstract_tr.to_csv(root_dir+subdir+''+' abstract_title_keys-terms_removed' ,index=False,header=False)
corpus_abstract_pure.to_csv(root_dir+subdir+'abstract_title pure',index=False,header=False)
# corpus_abstract_pure_tr.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' abstract_title-terms_removed',index=False,header=False)
# corpus_keywords.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' keywords',index=False,header=False)
# corpus_keywords_tr.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' keywords-terms_removed',index=False,header=False)
# corpus_keywords_index.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' keywords_index',index=False,header=False)
# corpus_keywords_index_tr.to_csv(root_dir+subdir+''+str(year_from)+'-'+str(year_to-1)+' keywords_index-terms_removed',index=False,header=False)
