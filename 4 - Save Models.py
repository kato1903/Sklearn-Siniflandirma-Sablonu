# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 23:13:51 2019

@author: Toprak
"""

import pickle

pickle.dump(LogisticRegression, open('LogisticRegression1', 'wb'))
pickle.dump(KNeighbors, open('KNeighbors', 'wb'))
pickle.dump(SupportVector, open('SupportVector', 'wb'))
pickle.dump(NaiveBayes, open('NaiveBayes', 'wb'))
pickle.dump(RandomForest, open('RandomForest', 'wb'))
pickle.dump(ExtreemRandom, open('ExtreemRandom', 'wb'))
pickle.dump(AdaBoost, open('AdaBoost', 'wb'))
pickle.dump(LDA, open('LDA', 'wb'))
pickle.dump(MLPClassifier, open('MLPClassifier', 'wb'))
pickle.dump(QDA, open('QDA', 'wb'))
pickle.dump(pca, open('pca', 'wb'))
pickle.dump(sc1, open('sc1', 'wb'))
pickle.dump(sc2, open('sc2', 'wb'))

pickle.dump(nuSVC, open('nuSVC', 'wb'))
pickle.dump(GradientBoostingClassifier, open('GradientBoostingClassifier', 'wb'))