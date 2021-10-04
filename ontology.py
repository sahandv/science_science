#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 22:38:55 2021

@author: github.com/sahandv
"""
import sys
import time
import gc
import os
import copy
import json
from multiprocessing import Pool
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
# import plotly.express as px
# from plotly.offline import plot
from random import randint
from scipy import spatial
# from treelib import Node, Tree
from sciosci.assets import text_assets as ta
from sciosci.assets import keyword_dictionaries as kd
# from gensim.parsing.preprocessing import strip_multiple_whitespaces
from networkx.drawing.nx_agraph import graphviz_layout,write_dot
import matplotlib.pyplot as plt



# directory = '/home/sahand/GoogleDrive/Data/Corpus/Dimensions All/clean/'
directory = '/home/sahand/GoogleDrive/Data/Corpus/Scopus new/clean/'

# file_name = 'cora deflemm'#corpus abstract-title - with n-grams'
file_name = 'data with abstract'#corpus abstract-title - with n-grams'
corpus = pd.read_csv(directory+file_name)
keywords = [item for sublist in corpus['DE'].values.tolist() for item in sublist.split(' | ')]
keywords = list(set(keywords))
keywords = pd.DataFrame(keywords,columns=['keyword'])
keywords.to_csv(directory+'keywords flat')
sample = corpus.sample(100)


stops = ['a','an','we','result','however','yet','since','previously','although','propose','proposed','this','...']

# from DEC.DEC_keras import DEC_simple_run
tqdm.pandas()
sample = keywords[:100]
# pd.options.mode.chained_assignment = None  # default='warn'

# =============================================================================
# Read Data
# =============================================================================
datapath = '/home/sahand/GoogleDrive/Data/Corpus/Taxonomy/'
relations = pd.read_csv(datapath+'CSO.3.3-with-labels.csv')
relations.a = relations.a.str.replace('@en .','')
relations.b = relations.b.str.replace('@en .','')
relations = relations[~relations['relation'].isin(['external-source-entity','link','type'])]

# =============================================================================
# Text pe-process
# =============================================================================
relations.a = relations.a.progress_apply(lambda x: ta.replace_british_american(x,kd.gb2us)) # Optional step
relations.b = relations.b.progress_apply(lambda x: ta.replace_british_american(x,kd.gb2us)) # Optional step
relations.a = relations.a.progress_apply(lambda x: ta.string_pre_processing(x,stemming_method='None',lemmatization='DEF',stop_word_removal=True,stop_words_extra=stops,verbose=False,download_nltk=False))
relations.b = relations.b.progress_apply(lambda x: ta.string_pre_processing(x,stemming_method='None',lemmatization='DEF',stop_word_removal=True,stop_words_extra=stops,verbose=False,download_nltk=False))
relations = relations[pd.notnull(relations.b)]
relations = relations[relations.a!='']
relations = relations[relations.b!='']
relations.to_csv(datapath+'CSO.3.3-with-labels-US-lem.csv',index=False)
# =============================================================================
# Group similar items according to the links
# =============================================================================
relations = pd.read_csv(datapath+'CSO.3.3-with-labels-US-lem.csv')

similar_terms = []
equality_relations = ['equal','readable-label','preferred-name-is']
relations_equality = relations[relations['relation'].isin(equality_relations)]

unique_a = list(set(relations_equality.a.values.tolist()))
unique_b = list(set(relations_equality.b.values.tolist()))
assert unique_a==unique_b, 'A and B are not equal, treat them differently and modify the code from here!'

equality_graph = nx.Graph()
for i,row in tqdm(relations_equality.iterrows(),total=relations_equality.shape[0]):
    equality_graph.add_edge(row['a'],(row['b']))
assert ~nx.is_connected(equality_graph), 'Graph is fully connected. It should not be! It means everything is equal. Check it out first.'
print('Connected components, AKA, Number of Concepts:',nx.number_connected_components(equality_graph))

unique_concepts = [list(c) for c in sorted(nx.connected_components(equality_graph), key=len)]
unique_concepts_labels = [c[0] for c in unique_concepts]

concept_ids = {}
for i,group in enumerate(unique_concepts):
    for concept in group:
        concept_ids[concept]=i
# =============================================================================
# Contruct Tree
# =============================================================================
relations_backup = relations.copy()
tree_relations = ['parent_of']
relations_tree = relations[relations['relation'].isin(tree_relations)]

tree_graph = nx.DiGraph()
for i,row in tqdm(relations_tree.iterrows(),total=relations_tree.shape[0]):
    tree_graph.add_edge(row['a'],(row['b']))
# assert nx.is_connected(tree_graph), 'Tree is not connected. Fix it first!'

# =============================================================================
# Contruct Tree - reverse
# =============================================================================
tree_graph = nx.DiGraph()
for i,row in tqdm(relations_tree.iterrows(),total=relations_tree.shape[0]):
    tree_graph.add_edge(row['b'],(row['a']))
# assert nx.is_connected(tree_graph), 'Tree is not connected. Fix it first!'


# =============================================================================
# Visualize
# =============================================================================
# write_dot(tree_graph,'tree.dot')

plt.title('draw_networkx-reverse')
plt.figure(figsize=(150, 150), dpi=150)
pos=graphviz_layout(tree_graph, prog='dot')
nx.draw(tree_graph, pos, with_labels=True, arrows=False)
plt.savefig('tree_nvlarge-reverse.png')



A = nx.drawing.nx_agraph.to_agraph(tree_graph)
A.layout('dot', args='-Nfontsize=5 -Nwidth=".2" -Nheight=".2" -Nmargin=0 -Gfontsize=5')
A.draw('tree_large_bubbles.png')
# =============================================================================
# Play with
# =============================================================================
cskids = tree_graph.neighbors('computer science')
cskids = [kid for kid in cskids]

parents = tree_graph.predecessors('passive tag')
parents = list(parents)

parents = tree_graph.predecessors('passive rfid')
parents = list(parents)

parents = tree_graph.predecessors('rfid reader')
parents = list(parents)

parents = tree_graph.predecessors('rfid system')
parents = list(parents)

# =============================================================================
# make parent table
# =============================================================================

# def get_parent(G,child_name,level):
#     level +=1
#     if level > 40:
#         return None
#     parents = list(G.predecessors(child_name))
#     # print(parents)
#     all_parents = []
#     if 'computer science' in parents:
#         return child_name
#     for parent in parents:
#         all_parents.append(get_parent(G,parent,level))
#     return all_parents


class Solution:
    def __init__(self,root='ROOT',limiter:int=30):
        self.results = []
        self.root = root
        self.limiter = limiter
        
    def get_parent(self,G,child_name,level=0):
        level +=1
        parents = list(G.predecessors(child_name))
        if self.root in parents:
            self.results.append(child_name)
        if level>=self.limiter:
            return 0
        for parent in parents:
            self.get_parent(G,parent,level)
    
solution = Solution('computer science')
solution.get_parent(tree_graph,'passive tag')
results = list(set(solution.results))

start = 4000
end = 6000

all_concepts = list(set(relations.a.values.tolist()+relations.b.values.tolist()))
if 'computer science' in all_concepts: all_concepts.remove('computer science')

results_all = {}
for concept in tqdm(all_concepts[start:end]):
    solution = Solution('computer science')
    solution.get_parent(tree_graph,'passive tag')
    results_all[concept] = list(set(solution.results))


output_address = '/home/sahand/GoogleDrive/Data/Corpus/Taxonomy/'+str(start)+' concept_parents lvl2'
with open(output_address, 'w') as json_file:
    json.dump(results_all, json_file)


# =============================================================================
# Dummy data test
# =============================================================================
G = nx.Graph()
G.add_node("ROOT")
for i in range(6):
    G.add_node("Child_%i" % i)
    G.add_node("Grandchild_%i" % i)
    G.add_node("Greatgrandchild_%i" % i)
    G.add_node("GGreatgrandchild_%i" % i)
    G.add_node("GGGreatgrandchild_%i" % i)

    # G.add_edge("ROOT", "Child_%i" % i)
    G.add_edge("Child_%i" % i, "Grandchild_%i" % i)
    G.add_edge("Grandchild_%i" % i, "Greatgrandchild_%i" % i)
    G.add_edge("Greatgrandchild_%i" % i, "GGreatgrandchild_%i" % i)
    G.add_edge("GGreatgrandchild_%i" % i, "GGGreatgrandchild_%i" % i)

G.add_edge("Grandchild_1", "Greatgrandchild_2")
G.add_edge("Child_1", "Greatgrandchild_3")
G.add_edge("Grandchild_1", "GGreatgrandchild_2")
G.add_edge("Grandchild_1", "GGreatgrandchild_3")

plt.title('draw_networkx')
pos =graphviz_layout(G, prog='dot')
nx.draw(G, pos, with_labels=True, arrows=True)
plt.savefig('result.svg')

solution = Solution('ROOT')
solution.get_parent(G,'GGGreatgrandchild_1')
set(solution.results)

    
    
delta_t = abs(model_backup.classifications['t']-model_backup.classifications['t'].values.max())

    
G = nx.Graph()
# G.add_nodes_from(lohg['nodes'])
G.add_edges_from(log['edges']['data'])
    
log=model_backup.temp
G = log['G']['data']

list(G.nodes)
    
    
    
    
#%%
# QUICK PARENT TABLE MAKER
import sys
import time
import gc
import os
import copy
import json
from multiprocessing import Pool
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import plot
from random import randint
from scipy import spatial
from treelib import Node, Tree
from sciosci.assets import text_assets as ta
from sciosci.assets import keyword_dictionaries as kd
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from networkx.drawing.nx_agraph import graphviz_layout,write_dot
import matplotlib.pyplot as plt
import itertools
datapath = '/home/sahand/GoogleDrive/Data/Corpus/Taxonomy/'
relations = pd.read_csv(datapath+'CSO.3.3-with-labels-US-lem.csv')



relations_backup = relations.copy()
tree_relations = ['parent_of']
relations_tree = relations[relations['relation'].isin(tree_relations)]

tree_graph = nx.DiGraph()
for i,row in tqdm(relations_tree.iterrows(),total=relations_tree.shape[0]):
    tree_graph.add_edge(row['a'],(row['b']))
# assert nx.is_connected(tree_graph), 'Tree is not connected. Fix it first!'

class Solution:
    def __init__(self,root='ROOT',limiter:int=100):
        self.results = []
        self.root = root
        self.limiter = limiter
    
    def get_parent(self,G,child_name,level=0):
        level +=1
        parents = list(G.predecessors(child_name))
        if self.root in parents:
            self.results.append(child_name)
        if level>=self.limiter:
            return 0
        for parent in parents:
            self.get_parent(G,parent,level)

solution = Solution('computer science')
solution.get_parent(tree_graph,'rfid')
results = list(set(solution.results))

list(tree_graph.predecessors('neural network'))

start = 0

all_concepts = list(set(relations.a.values.tolist()+relations.b.values.tolist()))
if 'computer science' in all_concepts: all_concepts.remove('computer science')

results_all = {}
for concept in tqdm(all_concepts[start:]):
    solution = Solution('computer science')
    solution.get_parent(tree_graph,'passive tag')
    results_all[concept] = list(set(solution.results))


output_address = '/home/sahand/GoogleDrive/Data/Corpus/Taxonomy/'+str(start)+' concept_parents lvl2'
with open(output_address, 'w') as json_file:
    json.dump(results_all, json_file)

# =============================================================================
# DFS
# =============================================================================
all_concepts = list(set(relations.a.values.tolist()+relations.b.values.tolist()))
if 'computer science' in all_concepts: all_concepts.remove('computer science')

def extraxt(link):
    if link[0]=='engineering':
        return 'engineering'
    if link[0]=='computer science':
        return link[1]
    return False

results_all = {}
for concept in tqdm(all_concepts):
    result = list(nx.edge_dfs(tree_graph,concept, orientation='reverse'))
    result = [extraxt(link) for link in result if extraxt(link)]
    if len(result)>0:
        results_all[concept] = result

output_address = '/home/sahand/GoogleDrive/Data/Corpus/Taxonomy/concept_parents lvl2 DFS'
with open(output_address, 'w') as json_file:
    json.dump(results_all, json_file)


# for start in range(2000,11000,2000):
#     print(start)
#     with open('/home/sahand/GoogleDrive/Data/Corpus/Taxonomy/'+str(start)+' concept_parents lvl2') as f:
#         data_0.update(json.load(f))

    
    
    
    
