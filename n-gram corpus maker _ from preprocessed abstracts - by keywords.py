#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:16:39 2020

@author: github.com/sahandv
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
# =============================================================================
# Init
# =============================================================================
print('\nInitializing...')
np.random.seed(50)

root_path = '/home/sahand/GoogleDrive/Data/Corpus/copyr_lemmatized_stopword_removed_thesaurus/'
data_abstracts = pd.read_csv(root_path+'1900-2019 abstract_title',names=['abstracts'])
data_years = pd.read_csv(root_path+'1900-2019 years')
data_keywords = pd.read_csv(root_path+'n-gram author keyword taxonomy.csv')

wanted_grams = [2,3,4,5,6,7] # Statistically, 5 seems to be a proper cutting point as the frequency table suggests. Refer to : "Get statsitic of n in n-grams of corpus" block in drafts.
periods = [[1990,2005],[2005,2008],[2008,2011],[2011,2014],[2014,2017],[2017,2019]]
thesaurus = []

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
    keywords_spaced = data_keywords_tmp.keywords.str.lower().str.strip().values.tolist()
    thesaurus.append(dict(zip(keywords_spaced,keywords_underscored)))
    idx+=1

# =============================================================================
# Replace by the thesaurus
# =============================================================================
print('\nApplying thesaurus...')
data_abstracts['abstracts_thesaurus'] = data_abstracts['abstracts']
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
    pd.DataFrame(corpora[i]).to_csv(root_path+'by period/n-gram by 2 repetition keywords/'+period+' abstract_title n_grams',index=False,header=False)

pd.DataFrame(data_abstracts['abstracts_thesaurus'].values.tolist()).to_csv(root_path+'1900-2019 n-gram by 2 repetition keywords',index=False,header=False)









