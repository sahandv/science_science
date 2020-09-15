#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:16:39 2020

@author: github.com/sahandv
"""
import sys
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

# =============================================================================
# Init
# =============================================================================
print('\nInitializing...')
np.random.seed(50)
sentence_replacer = False
root_path = '/home/sahand/GoogleDrive/Data/Corpus/AI 4k/copyr_deflem_stopword_removed_thesaurus May 28/'
data_abstracts = pd.read_csv(root_path+'by period/1900-2019 abstract_title',names=['abstracts'])
data_years = pd.read_csv(root_path+'by period/1900-2019 corpus years') #data_abstracts['year'] #
data_keywords = pd.read_csv(root_path+'../../Taxonomy/1980-2019 300k n-gram author keyword taxonomy - 95percentile and cleaned.csv')

wanted_grams = [2,3,4,5,6] # Statistically, 5 seems to be a proper cutting point as the frequency table suggests. Refer to : "Get statsitic of n in n-grams of corpus" block in drafts.
periods = [[1900,2020]]#,[1990,1995],[1995,2000],[2000,2005]]
thesaurus = []

data_years.hist(figsize=(15,5),bins=50,color='orange')#,cumulative=True)
data_keywords.grams.hist(figsize=(8,5),color='orange')
data_keywords[data_keywords['grams']>=6]
# =============================================================================
# Prepare keywords
# =============================================================================
print('\nPreparing keywords...')
data_keywords['grams'] = [len(x.split()) for x in data_keywords.keywords.values.tolist()]
data_keywords = data_keywords[data_keywords['count']>1]

# =============================================================================
# Make keyword dictionary/thesaurus for all wanted gram counts
# =============================================================================
print('\nPreparing thesaurus...')
idx = 0
for grams_count in tqdm(wanted_grams):
    data_keywords_tmp = data_keywords[data_keywords.grams>=wanted_grams[idx]]
    keywords_underscored = data_keywords_tmp.keywords.str.lower().str.strip().str.replace(' ','_').str.upper().values.tolist()
    keywords_underscored = [' '+x+' ' for x in keywords_underscored]
    keywords_spaced = data_keywords_tmp.keywords.str.lower().values.tolist() #.str.strip()
    thesaurus.append(dict(zip(keywords_spaced,keywords_underscored)))
    idx+=1

# =============================================================================
# Sentence replacer -- Fast -- if yes, skip the rest
# =============================================================================
if sentence_replacer is True:
    i = 0
    sentences = data_abstracts[i:i+100000].copy()
    del data_abstracts
    for thesaurus_gram in list(reversed(thesaurus)):
        rep = thesaurus_gram
        
        pattern = re.compile("|".join(rep.keys()))
        sentences = pd.DataFrame(sentences)
        sentences['sentence'] = sentences['sentence'].progress_apply(lambda x: pattern.sub(lambda m: rep[re.escape(m.group(0))], x)).str.lower()

        # sentences['sentence'] = sentences['sentence'].replace(thesaurus_gram, regex=True).values.tolist()
    
    pd.DataFrame(sentences).to_csv(root_path+'1900-2019 n-gram by 2 repetition keywords '+str(i),index=False,header=False)
    sys.exit('Did not continue to create normal corpus. If you want a corpus, set sentence_replacer to False at init section.')


# =============================================================================
# Replace by the thesaurus
# =============================================================================
print('\nApplying thesaurus...')
data_abstracts['abstracts_thesaurus'] = data_abstracts['abstracts']#abstracts

for thesaurus_gram in tqdm(list(reversed(thesaurus))):
    data_abstracts['abstracts_thesaurus'] = data_abstracts['abstracts_thesaurus'].replace(thesaurus_gram, regex=True)

# =============================================================================
# Divide by period
# =============================================================================
print('\nPreparing corpora by periods...')
corpora = []
period_names = []
for period in tqdm(periods):
    period_names.append(str(period[0])+'-'+str(period[1]-1))
    corpora.append(data_abstracts[(data_years['year']>=period[0]) & (data_years['year']<period[1])]['abstracts_thesaurus'].values.tolist())

# =============================================================================
# Save to disk
# =============================================================================
print('\nWriting to disk...')
for i,period in tqdm(enumerate(period_names)):
    pd.DataFrame(corpora[i]).to_csv(root_path+'by period/n-gram by 6 repetition keywords/'+period+' abstract_title',index=False,header=False)

pd.DataFrame(data_abstracts['abstracts_thesaurus'].values.tolist()).to_csv(root_path+'1900-2019 n-gram by 6 repetition keywords',index=False,header=False)







