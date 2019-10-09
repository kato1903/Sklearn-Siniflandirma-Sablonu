# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 23:04:54 2019

@author: Toprak
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA(n_components = 2)

x = pca.fit_transform(x)

sc2 = StandardScaler()

x = sc2.fit_transform(x)