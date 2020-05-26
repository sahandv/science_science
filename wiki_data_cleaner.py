#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:28:36 2019

@author: github.com/sahandv
"""
import sys, os, time
import platform
from pathlib import Path
import sys
import gc
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from stat import S_ISREG, ST_CTIME, ST_MODE
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sciosci.assets import keyword_assets as kw
from sciosci.assets import keyword_dictionaries as kd

tqdm.pandas()
nltk.download('wordnet')
nltk.download('punkt')

# =============================================================================
# Input
# =============================================================================
pages_dir = str(Path.home())+'/GoogleDrive/Data/Corpus/AI Wiki Classifications/applications/'
root_dir = str(Path.home())+'/GoogleDrive/Data/Corpus/AI Wiki Classifications/applications/'
subdir = 'clean/'

# =============================================================================
# Read data and Initialize
# =============================================================================

print("\nSearching for Wiki texts...")
dir_path = pages_dir
data = (os.path.join(dir_path, fn) for fn in os.listdir(dir_path))
data = ((os.stat(path), path) for path in data)
data = ((stat[ST_CTIME], path) for stat, path in data if S_ISREG(stat[ST_MODE]))

names = []
files = []
for cdate, path in sorted(data):
    print('   - ', time.ctime(cdate), os.path.basename(path),int(os.path.getsize(path)/1000000),'MB')
    files.append(path)
    names.append(os.path.basename(path))

print("Will process these files.")
try:
    input("Press enter to accept and continue...")
except SyntaxError:
    pass

for file_index,file in enumerate(files):
    data_path_rel = file
    print("\nPreparing files...")
    stops = ['a','an','we','result','however','yet','since','previously','although','propose','proposed','this']
    nltk.download('stopwords')
    stop_words = list(set(stopwords.words("english")))+stops
    
    data_full_relevant = pd.read_csv(data_path_rel,sep=';;;;',names=['paragraph'])    
    gc.collect()
    
    # =============================================================================
    # Initial Pre-Processing : 
    #   Following tags requires WoS format. Change them otherwise.
    # =============================================================================
    data_filtered = data_full_relevant.copy()
    data_filtered = data_filtered[pd.notnull(data_filtered['paragraph'])]
    
    # Remove numbers from abstracts to eliminate decimal points and other unnecessary data
    data_filtered['paragraph'] = data_filtered['paragraph'].progress_apply(lambda x: kw.find_and_remove_c(x) if pd.notnull(x) else np.nan).str.lower()
    data_filtered['paragraph'] = data_filtered['paragraph'].progress_apply(lambda x: kw.find_and_remove_term(x,'et al.') if pd.notnull(x) else np.nan)
    data_filtered['paragraph'] = data_filtered['paragraph'].progress_apply(lambda x: kw.find_and_remove_term(x,'eg.') if pd.notnull(x) else np.nan)
    data_filtered['paragraph'] = data_filtered['paragraph'].progress_apply(lambda x: kw.find_and_remove_term(x,'ie.') if pd.notnull(x) else np.nan)
    data_filtered['paragraph'] = data_filtered['paragraph'].progress_apply(lambda x: kw.find_and_remove_term(x,'vs.') if pd.notnull(x) else np.nan)
    data_filtered['paragraph'] = data_filtered['paragraph'].progress_apply(lambda x: kw.find_and_remove_term(x,'ieee') if pd.notnull(x) else np.nan)
    data_filtered['paragraph'] = data_filtered['paragraph'].progress_apply(lambda x: kw.find_and_remove_term(x,'fig.','figure') if pd.notnull(x) else np.nan)
    data_filtered['paragraph'] = data_filtered['paragraph'].progress_apply(lambda x: kw.find_and_remove_term(x,'-',' ') if pd.notnull(x) else np.nan)
    # data_filtered['paragraph'] = data_filtered['paragraph'].progress_apply(lambda x: kw.find_and_remove_term(x,'(',' ') if pd.notnull(x) else np.nan)
    # data_filtered['paragraph'] = data_filtered['paragraph'].progress_apply(lambda x: kw.find_and_remove_term(x,')',' ') if pd.notnull(x) else np.nan)
    data_filtered['paragraph'] = data_filtered['paragraph'].progress_apply(lambda x: kw.find_and_remove_term(x,'—',' ') if pd.notnull(x) else np.nan)
    data_filtered['paragraph'] = data_filtered['paragraph'].progress_apply(lambda x: kw.find_and_remove_term(x,'_',' ') if pd.notnull(x) else np.nan)
    data_filtered['paragraph'] = data_filtered['paragraph'].progress_apply(lambda x: kw.find_and_remove_term(x,',',' ') if pd.notnull(x) else np.nan)
    
    # gc.collect()
    text = []
    for line in tqdm(data_filtered['paragraph'].values.tolist()):
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        for number in numbers:
            line = kw.find_and_remove_term(line,number)
        text.append(line)
    data_filtered['paragraph'] = text.copy()
    del  text 
    
    data_fresh = data_filtered['paragraph'].str.lower()
    
    print("\nSentence extraction")
    sentences = []
    years = []
    indices = []
    for row in tqdm(data_fresh.to_list()):
        abstract_sentences = re.split('\. |\? |\\n',row)
        sentences.extend(abstract_sentences)
    
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
    
    print("\nWriting to disk")
    sentence_df = pd.DataFrame(sentences,columns=['sentence'])
    sentence_df.to_csv(root_dir+subdir+names[file_index]+'',index=False,header=True)
