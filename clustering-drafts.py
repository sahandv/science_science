#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 01:40:25 2021

@author: sahand
"""


import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout,write_dot
import matplotlib.pyplot as plt
import itertools
from collections import Counter


tqdm.pandas()

classifications = pd.DataFrame({'t':[0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2],
                                'class':[0,3,1,1,1,1,0,1,0,3,1,1,0,1,0,1,1,0,3,2,0,1]})
len(classifications[classifications['t']==0]['class'].value_counts().keys())

classifications_new = classifications[classifications['class']==1]

for i,row in classifications_new[['t','class']].iterrows():
    classifications['class'][i] = row['t']*100
    print(i)

to_ignore = [3]
t=2
classifications_populations_old = classifications[classifications['t']==t-1][['class']].value_counts() # we should consider to_ignore for correct total population 
classifications_populations_old = pd.DataFrame(classifications_populations_old,columns=['population'])
classifications_populations_old.reset_index(inplace=True)

classifications_populations = classifications[classifications['t']==t][['class']].value_counts() # to_ignore is already eliminated in previous step, if any
classifications_populations = pd.DataFrame(classifications_populations,columns=['population'])
classifications_populations.reset_index(inplace=True)


classifications_populations['class'].values.tolist()
c=0

initial_radius = {1:1.2,2:4,5:1}
initial_radius[3]

to_split = {1:1.2,2:4,5:1}



corpus_data = pd.read_csv('/home/sahand/GoogleDrive/Data/Corpus/Dimensions All/clean/abstract_title method_b_3')
full_data = pd.read_csv('/home/sahand/GoogleDrive/Data/Corpus/Dimensions All/clean/data with abstract')
categories = full_data['FOR'].str.split(';;;')
categories = [[c.split(':::')[1] for c in category] for category in tqdm(categories)]
categories_full = [list(set([c.split()[0] for c in category if len(c.split()[0])>2] )) for category in tqdm(categories)]
categories_initials = [list(set([c[:2] for c in category] )) for category in tqdm(categories_full)]

wanted_full = set(['0801','0802','0803','0804','0805','0806','0807','0899',
                   '1004','1005','1006','1007'])
categories_full = [list(set(category).intersection(wanted_full)) for category in tqdm(categories_full)]





sample = categories[:100]
sample = categories_initials[:100]

categories_initials = [','.join(category) for category in tqdm(categories_initials)]
categories_full = [','.join(category) for category in tqdm(categories_full)]
full_data['FOR_initials'] = categories_initials
full_data['FOR_full'] = categories_full

full_data.to_csv('/home/sahand/GoogleDrive/Data/Corpus/Dimensions All/clean/data with abstract')


wanted = set(['08','10'])

full_data['mask'] = full_data['FOR_initials'].progress_apply(lambda x: True if len(set(x.split(',')).intersection(wanted))>0 else False)

sample = full_data.sample(10)


full_data_masked = full_data[full_data['mask']==True]
full_data_masked.drop('mask',axis=1,inplace=True)
full_data_masked.to_csv('/home/sahand/GoogleDrive/Data/Corpus/Dimensions All/clean/data with abstract - tech and comp - with subcategories')


index = pd.Index(['a', '1', '2', '2', '4', np.nan])
counts = index.value_counts()
counts['2']

mylist = ['a', '1', '2', '2', '4', np.nan]

'a' in mylist





roots = [['d','c','e'],['c','d'],['d','c'],['d','e'],['a','b'],['b','a','f'],['b','f','a'],['a','c']]
nodes = list(set(list(itertools.chain.from_iterable(roots))))+['h']
edges = list(itertools.chain.from_iterable([[list(set(list(x))) for x in list(itertools.combinations(sets, 2))] for sets in roots]))


G = nx.MultiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

plt.title('draw_networkx')
pos =graphviz_layout(G, prog='dot')
nx.draw(G, pos, with_labels=True, arrows=True)

for i in G.nodes:
    print(i, G.edges(i))
    
pos = nx.kamada_kawai_layout(G)
nx.draw_networkx_nodes(G, pos, node_color = 'r', node_size = 100, alpha = 1)
ax = plt.gca()
for e in G.edges:
    ax.annotate("",
                xy=pos[e[0]], xycoords='data',
                xytext=pos[e[1]], textcoords='data',
                arrowprops=dict(arrowstyle="->", color="0.5",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*e[2])
                                ),
                                ),
                )
plt.axis('off')
plt.show()


to_delete = [list(x) for x in list(itertools.combinations(nodes, 2))]
G.remove_edges_from(to_delete)



edges_counts = list(G.subgraph(c).number_of_edges() for c in nx.connected_components(G))
edges_counts = [x for x in edges_counts if x>2]
to_split = [x for x in list(itertools.combinations(edges_counts, 2)) if min(x[0],x[1])/max(x[0],x[1])>1/2]
list(itertools.chain.from_iterable(to_split))

list(G.nodes())[0]

# =============================================================================
# 
# =============================================================================

class_centroid_proposal = {1:[1,2],2:[1,3],3:[4,5],4:[]}
class_centroid_proposal = {k:v for k,v in class_centroid_proposal.items() if len(v)>=2}
to_split = {1:1,2:3,3:2,4:1}

while len(class_centroid_proposal)>0:
    print("Cluster split votes are as follows:")
    print(to_split)
    
    user_input = input("Which cluster you want to re-cluster? (N: none, A: All, or from: "+str([k for k,v in class_centroid_proposal.items() if len(v) >= 2])+")\n")
    if user_input=='N':
        class_centroid_proposal = {}
    elif user_input=='A':
        class_centroid_proposal = {}
    else:
        print(' -  - sub clustering cluster '+str(user_input))
        to_recluster = int(user_input)
        del class_centroid_proposal[to_recluster]
    
    





# =============================================================================
# 
# =============================================================================
from gensim.models import FastText as fasttext_gensim
from scipy import spatial
import numpy as np
import time


gensim_model_address_AI = '/home/sahand/GoogleDrive/Data/Corpus/Dimensions All/models/Fasttext/gensim381-w10/FastText100D-dim-scopus-update-window10.model'
gensim_model_address = '/home/sahand/GoogleDrive/Data/Corpus/Dimensions All/models/Fasttext/gensim381/FastText100D-dim-scopus-update.model'
model_AI = fasttext_gensim.load(gensim_model_address)
model = fasttext_gensim.load(gensim_model_address)

start = time.time()
vec_a = np.array([model_AI.wv['machine'],model_AI.wv['learning']]).mean(axis=0)
vec_b = np.array([model_AI.wv['deep'],model_AI.wv['learning']]).mean(axis=0)#(model_AI['data']+model_AI['science'])
result = 1-spatial.distance.cosine(vec_a, vec_b)
end = time.time()


start = time.time()
spatial.distance.cosine(all_vecs[0],ontology_dict['machine learning']['vector'])
end = time.time()
spent_time =float(end-start)


vec_a = np.array([model.wv['machine'],model.wv['learning']]).mean(axis=0)
vec_b = np.array([model.wv['deep'],model.wv['learning']]).mean(axis=0)#(model_AI['data']+model_AI['science'])
1-spatial.distance.cosine(vec_a, vec_b)

# =============================================================================
# work with sets and etc. to remove duplications and etc.
# =============================================================================

edges = list(itertools.chain.from_iterable([[list(set(list(x))) for x in list(itertools.combinations(list(set(sets)), 2))] for sets in tmp['roots']['data']])) # make pairs, hence the links

to_inspect = ['a','b','c']
to_ignore = [('a','b')]

neighbours = [[1, 2], [4], [5, 6, 2], [2, 1], [3], [4]]
neighbours= [set(i) for i in neighbours]

to_ignore = [[1,2]]
to_ignore = [set(i) for i in to_ignore]

neighbours_new = []
for elem in neighbours:
    if elem not in to_ignore:
        neighbours_new.append(elem)

neighbours = list(list(n) for n in neighbours)

# prints [[1, 2], [4], [5, 6, 2], [3]]


