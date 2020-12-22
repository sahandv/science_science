#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 11:44:42 2020

@author: sahand
"""

import networkx as nx
import numpy as np
import random

def graph_walks(graph:nx.Graph,seed=None,walk_length:int = 5,number_of_walks:int=1000):
    """
    Parameters
    ----------
    graph : nx.Graph
        Input graph.
    seed : TYPE, optional
        Numpy seed to reproduce results. The default is None.
    walk_length : int, optional
        The default is 5.
    number_of_walks : int, optional
        The default is 1000.

    Returns
    -------
    walks : TYPE
        DESCRIPTION.

    """
    if seed != None:
        np.random.seed(seed)
        
    walks = []
    for node in graph.nodes():
        for _ in range(number_of_walks):
            walks.append(node_random_walk(graph,node,walk_length,seed))
    return walks
            
def node_random_walk(graph:nx.Graph, node:int, walk_length:int = 5,seed=None):
    """
    Parameters
    ----------
    graph : nx.Graph
        Input graph.
    node : int
        Starting node name/index.
    walk_length : int, optional
        The default is 5.
    seed : int, optional
        Numpy seed to reproduce results. The default is None.

    Returns
    -------
    walk : list
        A list of the path of walked nodes.

    """
    if seed != None:
        np.random.seed(seed)
    
    walk = [str(node),]
    target_node = node
    for _ in range(walk_length):
        neighbors = list(nx.all_neighbors(graph, target_node))
        target_node = random.choice(neighbors)
        walk.append(str(target_node))
        
    return walk