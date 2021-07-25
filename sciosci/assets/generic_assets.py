#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:49:39 2019

@author: github.com/sahandv
"""

def general_initialize(ignore_py_version = False,
                      ignore_scopus_config = False):
    import sys
    import platform
    from pathlib import Path
    
    if platform.system() != 'Linux':
        print("\n * It is recommended to use a Linux system.")
        
    if ignore_py_version is False:
        try:
            assert sys.version_info >= (3, 5)
        except AssertionError:
            sys.exit("\n * Please use Python 3.5 +")
    
    try:
        import mmap
    except ImportError:
        sys.exit("\n * Please install mmap.")

    try:
        import tqdm
    except ImportError:
        sys.exit("\n * Please install tqdm.")
    
    try:
        import pandas
    except ImportError:
        sys.exit("\n * Please install pandas.")
    
    try:
        import nltk
    except ImportError:
        sys.exit("\n * Please install nltk.")    
        
    try:
        import sklearn
    except ImportError:
        sys.exit("\n * Please install scikit-learn.")    
        
    try:
        import scipy
    except ImportError:
        sys.exit("\n * Please install scipy.")    
        
    try:
        import seaborn
    except ImportError:
        sys.exit("\n * Please install seaborn.")    
        
    try:
        import wordcloud
    except ImportError:
        sys.exit("\n * Please install wordcloud.")   
        
    try:
        import matplotlib
    except ImportError:
        sys.exit("\n * Please install matplotlib.")   
        
    try:
        import PIL
    except ImportError:
        sys.exit("\n * Please install pillow.")
                
    try:
        import tabulate
    except ImportError:
        sys.exit("\n * Please install tabulate.")
    
    try:
        import gensim
    except ImportError:
        sys.exit("\n * Please install 'gensim' and download pretrained models from http://nlp.stanford.edu/data/ .")


