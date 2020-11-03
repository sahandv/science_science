#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 17:59:55 2020

@author: sahand
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
# Create data
X = np.arange(0, 10, 1)
Y = X + 5 * np.random.random((5, X.size))
 
# There are 4 types of baseline we can use:
baseline = ["zero", "sym", "wiggle", "weighted_wiggle"]
 
# Let's make 4 plots, 1 for each baseline
for n, v in enumerate(baseline):
   if n<3 :
      plt.tick_params(labelbottom='off')
   plt.subplot(2 ,2, n + 1)
   plt.stackplot(X, *Y, baseline=v)
   plt.title(v)
   plt.axis('tight', size=0.2)
