# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 23:30:40 2019

@author: Toprak
"""

import pickle
import pandas as pd
import numpy as np

df = pd.read_csv('Veri.csv')
df = pd.read_excel('veri.xlsx')

x = df.iloc[:,6:18].values
y = df.iloc[:,5].values


LogisticRegression = pickle.load(open('LogisticRegression', 'rb'))
KNeighbors = pickle.load(open('KNeighbors', 'rb'))
SupportVector = pickle.load(open('SupportVector', 'rb'))
NaiveBayes = pickle.load(open('NaiveBayes', 'rb'))
RandomForest = pickle.load(open('RandomForest', 'rb'))
ExtreemRandom = pickle.load(open('ExtreemRandom', 'rb'))
AdaBoost = pickle.load(open('AdaBoost', 'rb'))
LDA = pickle.load(open('LDA', 'rb'))
MLPClassifier = pickle.load(open('MLPClassifier', 'rb'))
QDA = pickle.load(open('QDA', 'rb'))
pca = pickle.load(open('pca', 'rb'))
sc1 = pickle.load(open('sc1', 'rb'))
sc2 = pickle.load(open('sc2', 'rb'))
#LinearRegression = pickle.load(open('LinearRegression', 'rb'))

nuSVC = pickle.load(open('nuSVC', 'rb'))
GradientBoostingClassifier = pickle.load(open('GradientBoostingClassifier', 'rb'))

from sklearn.preprocessing import StandardScaler

x = sc1.transform(x)
x = pca.transform(x)
x = sc2.transform(x)

y_pred1 = LogisticRegression.predict(x)
y_pred2 = KNeighbors.predict(x)
y_pred3 = SupportVector.predict(x)
y_pred4 = NaiveBayes.predict(x)
y_pred6 = RandomForest.predict(x)
y_pred7 = ExtreemRandom.predict(x)
y_pred8 = AdaBoost.predict(x)
y_pred9 = LDA.predict(x)
y_pred10 = MLPClassifier.predict(x)
y_pred11 = QDA.predict(x)



y_predP1 = LogisticRegression.predict_proba(x)
y_predP2 = KNeighbors.predict_proba(x)
y_predP3 = SupportVector.predict_proba(x)
y_predP4 = NaiveBayes.predict_proba(x)
y_predP6 = RandomForest.predict_proba(x)
y_predP7 = ExtreemRandom.predict_proba(x)
y_predP8 = AdaBoost.predict_proba(x)
y_predP9 = LDA.predict_proba(x)
y_predP10 = MLPClassifier.predict_proba(x)
y_predP11 = QDA.predict_proba(x)
y_predP12 = nuSVC.predict_proba(x)
y_predP13 = GradientBoostingClassifier.predict_proba(x)