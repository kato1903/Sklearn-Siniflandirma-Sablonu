# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 22:56:38 2019

@author: Toprak
"""

import pandas as pd

df = pd.read_csv('veri.csv')
df = pd.read_excel('veri.xlsx')

x = df.iloc[:,6:18].values
y = df.iloc[:,5].values

