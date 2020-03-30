#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:34:24 2019

@author: github.com/sahandv
"""
import numpy as np

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(glove_small))])
        else:
            self.dim=0
            
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    
# and a tf-idf version of the same
class TfidfEmbeddingVectorizer(object):
    
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(glove_small))])
        else:
            self.dim=0
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
                
                
def topic_label_maker(terms,scores,ratio=2,prefix='',limit=None):
    """
    Get 2 topic modeling tables of term-topic as 2 dataframes with identical 
    dimensions
    
    Outputs a list of the most important keywords for each column based on the 
    scores each column (topic/cluster) should be named sequentialy, starting 
    from 0, as string ratio: the threshold of keyword score ratio to the top 
    keyword score
    
    For the method to work properly, make sure the column names for scores and 
    terms are int
    ----------
    
    Parameters
    ----------
    terms : Pandas Dataframe of size n*m
        DESCRIPTION.
    scores : Pandas Dataframe of size n*m
        DESCRIPTION.
    ratio : TYPE, optional
        DESCRIPTION. The default is 2.
    prefix : TYPE, optional
        DESCRIPTION. The default is ''.
    limit : TYPE, optional
        DESCRIPTION. The default is None.

    Returns List
    -------
    None.
    
    -------
    For the method to work properly, make sure the column names for scores and terms are int

    """

    import pandas as pd
    from tqdm import tqdm
    
    labels = []
    for topic_index,topic_terms in tqdm(terms.T.iterrows(),total=terms.shape[1]):
        topic_label_terms = []
        counter = 0
        for term_index, term in topic_terms.iteritems():
            if scores[str(topic_index)][term_index] < scores[str(topic_index)][0]/ratio or term=='':
                break
            else:
                if limit is not None and limit <= counter:
                    break
                else:
                    topic_label_terms.append(term)
                    counter +=1
        labels.append(", ".join(topic_label_terms))
    
    labels = [prefix+x for x in labels]
    return labels

                
def draw_graph_from_csr(sparse_mat,size=(10, 10),dpi = 100,fname="graph_out.png",
                        save=False,labels=False,alpha=0.8,node_color=None,
                        node_color_palette=None,edge_color='c',groups=None,
                        min_degree=3,pos_g='spring_layout',rad=3,node_frequency=None,
                        node_size_variation=0,node_size_minimum=20,edge_weight_thresh=False):
    """
    
    Parameters
    ----------
    sparse_mat : TYPE
        DESCRIPTION.
    size : TYPE, optional
        DESCRIPTION. The default is (10, 10). Figue size
    dpi : TYPE, optional
        DESCRIPTION. The default is 100.
    fname : TYPE, optional
        DESCRIPTION. The default is "graph_out.png".
    save : TYPE, optional
        DESCRIPTION. The default is False. Save the figure to disk.
    labels : TYPE, optional
        DESCRIPTION. The default is False.
    alpha : TYPE, optional
        DESCRIPTION. The default is 0.8.
    node_color : TYPE, optional
        DESCRIPTION. The default is None.
    node_color_palette : 2D list, optional
        DESCRIPTION. The default is None. RGB colors in range of 0 to 1. Can be created by color_palette_maker. (number of rows == number of gropus)
    edge_color : TYPE, optional
        DESCRIPTION. The default is 'c'.
    groups : TYPE, optional
        DESCRIPTION. The default is None.
    min_degree : TYPE, optional
        DESCRIPTION. The default is 3. Values <3 means all isolated nodes will be removed
    pos_g : TYPE, optional
        DESCRIPTION. The default is 'spring_layout'.
    rad : TYPE, optional
        DESCRIPTION. The default is 3.
    node_frequency : TYPE, optional
        DESCRIPTION. The default is None.
    node_size_variation : TYPE, optional
        DESCRIPTION. The default is 0. A value between 0 to 10, 0 means no variation.
    node_size_minimum : TYPE, optional
        DESCRIPTION. The default is 20. Minimum size for the node in pixels.
    edge_weight_thresh : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """

    import networkx as nx
    import matplotlib.pyplot as plt
    import pandas as pd
    from tqdm import tqdm
    
    print('Creating POS')
    G = nx.from_scipy_sparse_matrix(sparse_mat, create_using=nx.MultiGraph())
    if pos_g == 'spring_layout':
        pos = nx.spring_layout(G)
    if pos_g == 'circular_layout':
        pos = nx.circular_layout(G)
    if pos_g == 'shell_layout':
        pos = nx.shell_layout(G)
    if pos_g == 'bipartite_layout':
        pos = nx.bipartite_layout(G,labels.items())
    if pos_g == 'fruchterman_reingold_layout':
        pos = nx.fruchterman_reingold_layout(G)
    if pos_g == 'spectral_layout':
        pos = nx.spectral_layout(G)
    
    print('Separating cluster positions by POS')
    if groups!=None:
        angs = np.linspace(0, 2*np.pi, 1+len(node_color_palette))
        repos = []
        for ea in angs:
            if ea > 0:
                repos.append(np.array([rad*np.cos(ea), rad*np.sin(ea)]))
        counter = 0
        for ea in pos.keys():      
            pos[ea] += (repos[groups[counter]]/100)
            counter +=1

    print('Removing low degree nodes')
    remove_from = [ind for ind, degree in enumerate(dict(G.degree).values()) if degree < min_degree]
    print('removing following nodes due to low degree',remove_from)
    G.remove_nodes_from(remove_from)
    [labels.pop(key) for key in remove_from]
    
    print("Coloring nodes")
    if node_color_palette!= None:
        node_color = cluster_color_mapper(groups,node_color_palette)
        color_groups_map = pd.DataFrame({'color':node_color,'groups':groups})
        if node_frequency!=None:
            color_groups_map['frequency'] = node_frequency
        color_groups_map = color_groups_map.drop(remove_from,axis=0)
        node_color = color_groups_map['color'].tolist()
    else:
        if node_color==None:
            node_color = 'r'
    
    
    if edge_weight_thresh is not False:
        print('> Filtering edges')
        edge_data = list(G.edges.data('weight', default=1))
        if edge_weight_thresh is None:
            edge_data_np = np.array(edge_data).T[2]
            edge_data_max = max(edge_data_np)
            edge_weight_thresh = np.exp(np.log(edge_data_max)/2)
            
        remove_edge_from = []
        for edge in tqdm(edge_data,total=len(edge_data)):
            if edge[2]<edge_weight_thresh:
                remove_edge_from.append((edge[0],edge[1]))
        G.remove_edges_from(remove_edge_from)
    
    if node_frequency!=None:
        print('> Resizing nodes based on frequency')
        node_size = np.power(np.log(np.array(color_groups_map['frequency'].tolist())),node_size_variation)+node_size_minimum
        print("> Calculating node alpha values")
        max_weight = max(node_frequency)
        node_alpha = [np.log(x)/np.log(max_weight) for x in color_groups_map['frequency'].tolist()]
    else:
        node_size = 100
        node_alpha = 0.8
        
    for breakpoint_idx,alpha in enumerate(node_alpha):
        if alpha < 0.35:
            break
        
    for idx in range(breakpoint_idx,len(node_alpha)):
        labels[list(labels.keys())[idx]] = ''
    

    
# =============================================================================
#     node_alpha_1 = node_alpha[:breakpoint_idx]
#     node_alpha_2 = 0.05
#     node_color_2 = node_color[breakpoint_idx:]
#     node_size_2 = node_size[breakpoint_idx:]
#        
#     print('> Drawing')
#     fig = plt.figure(3,figsize=size,facecolor='w', edgecolor='k')
#     idx = 0
#     for alpha_val in node_alpha_1:
#         nx.draw_networkx_nodes(G ,pos=pos, nodelist=[list(pos.keys())[idx]],node_size=node_size[idx], node_color=node_color[idx],alpha=alpha_val)
#         idx+=1
#         
#     print('> Drew',idx,'nodes. Drawing the rest now.')
#     nx.draw_networkx_nodes(G ,pos=pos, nodelist=list(pos.keys())[breakpoint_idx:], node_size=node_size_2, node_color=node_color_2,alpha=node_alpha_2)
#     print('> Drawing the edges now.')
#     nx.draw_networkx_edges(G ,pos=pos, edge_color='c',alpha=alpha)
#     print('> Drawing the lables now.')
#     nx.draw_networkx_labels(G ,pos=pos, node_size=node_size, with_labels=True, labels=labels)
#     if save is True:
#         fig.savefig(fname, dpi=dpi)
#     plt.show()
# =============================================================================
    
    print('Drawing')
    fig = plt.figure(3,figsize=size,facecolor='w', edgecolor='k')
    if labels is False:
        nx.draw(G, node_size=node_size , with_labels=True,alpha=0.6,node_color=node_color,edge_color=edge_color,pos=pos)
    else:
        nx.draw(G, node_size=node_size , with_labels=True, labels=labels,alpha=0.6,node_color=node_color,edge_color=edge_color,pos=pos)
    if save is True:
        fig.savefig(fname, dpi=dpi)
    plt.show()



def distance_matrix_from_points(points):
    """
    

    Parameters
    ----------
    points : 2D List
        List of vectors or 2D vector (each row is a vector).
        Each row is a vector for a record.
        
        Example:
            points=|0.2 0.1 0.3 0.5 |
                    |0.6 0.7 0.4 0.0 |
                    |0.8 0.2 0.3 0.9 |
                    |0.1 0.1 0.9 0.1 |

    Returns distance matrix
    -------
    None.

    """

    from scipy import spatial
    
    distances = []
    for row in points:
        distances_row = []
        for row_inner in points:
            distances_row.append(spatial.distance.cosine(row,row_inner))
        distances.append(distances_row)
    distances = np.array(distances)
    return distances



def abbreviator(string):
# =============================================================================
#     Create an abbreviation of the string to shorten it
# =============================================================================
    import re
    split_string = re.split(' |-',string)
    new_string = []
    for split in split_string:
        if split != '':
            new_string.append(split[0])
    return ''.join(new_string).upper()

def caption_maker_by_terms_df(df_terms,df_values,variation=1.5):
# =============================================================================
#     Gets terms and their scores from LDA output or similar outputs
#     This method will decide a name for these topics (or clusters) based on keyword value 
# =============================================================================
    names = []
    for index, row in df_values.iterrows():
        if df_values[int(index):int(index)+1][0].values.tolist()[0]/df_values[int(index):int(index)+1][1].values.tolist()[0] > variation:
            names.append(df_terms[int(index):int(index)+1][0].values.tolist()[0])
        else:
            names.append(df_terms[int(index):int(index)+1][0].values.tolist()[0]+' '+df_terms[int(index):int(index)+1][1].values.tolist()[0])
    return names


def color_palette_maker(n_colors,alpha=None):
# =============================================================================
#    get number of colors you want and the method will make colors
# =============================================================================
    colors = []
    for item in range(n_colors):
        if alpha is None:
            colors.append(list(np.random.choice(range(256), size=3)/256))
        else:
            colors.append(list(np.random.choice(range(256), size=3)/256)+[alpha])
    return colors


def cluster_color_mapper(clusters,color_palette=None):
# =============================================================================
#    clusters: a liste of cluster numbers for the nodes. Cluster numbers must be integer.
#    color_palette: a palette created by color_palette_maker(n_colors)
# =============================================================================
    if color_palette == None:
        print('Please use advanced_assets.color_palette_maker(n_colors) method to make a color palette and provide the palette to this function.')
        return False
    colors = []
    for item in clusters:
        colors.append(color_palette[item])
    return colors
        
def fancy_dendrogram(*args, **kwargs):
# =============================================================================
#     Taken from: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
#     Most credits to original author.
# ===========================================================================60==
    import scipy.cluster.hierarchy as sch
    import matplotlib.pyplot as plt

    max_d = kwargs.pop('max_d', None)
    
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)
    
    if 'figsize' in kwargs:
        figsize = kwargs.pop('figsize', None)
        plt.figure(figsize=figsize)
        
    ddata = sch.dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata



def bert_pair_gen(corpus):
    import re
    from tqdm import tqdm
    
    sentence_pairs = []
    for idx,row in tqdm(enumerate(corpus),total=len(corpus)):
        sentences = re.split('\. |\? |\n',row)
        sentences_num = len(sentences)
        for sidx,sentence in enumerate(sentences):
            if sidx==sentences_num-1:
                break
            sentence_pairs.append('[CLS] '+sentences[sidx]+' [SEP] '+sentences[sidx+1]+' [SEP]')
    return sentence_pairs


def bert_sentence_gen(corpus):
    import re
    from tqdm import tqdm
    
    sentence_list = []
    for idx,row in tqdm(enumerate(corpus),total=len(corpus)):
        sentences = re.split('\. |\? |\n',row)
        for sidx,sentence in enumerate(sentences):
            sentence_list.append('[CLS] '+sentences[sidx]+' [SEP]')
    return sentence_list


def compute_pc(X,npc=1):
# =============================================================================
#     Compute the principal components. 
#          DO NOT MAKE THE DATA ZERO MEAN!
#    
#     :param X: X[i,:] is a data point
#     :param npc: number of principal components to remove
#     :return: component_[i,:] is the i-th pc
# =============================================================================
    from sklearn.decomposition import TruncatedSVD
    
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X, npc=1):
# =============================================================================
#     Remove the projection on the principal components
#     :param X: X[i,:] is a data point
#     :param npc: number of principal components to remove
#     :return: XX[i, :] is the data point after removing its projection
# =============================================================================
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX



