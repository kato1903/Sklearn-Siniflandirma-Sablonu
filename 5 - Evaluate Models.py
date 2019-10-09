# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 23:16:40 2019

@author: Toprak
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
scores = cross_val_score(LogisticRegression, x, y, cv=5)

sumscores = 0

print("LogisticRegression " + str(scores.mean()))

sumscores += scores.mean()

scores = cross_val_score(KNeighbors, x, y, cv=5)

print("KNeighbors " + str(scores.mean()))

sumscores += scores.mean()

scores = cross_val_score(SupportVector, x, y, cv=5)

print("SupportVector " + str(scores.mean()))

sumscores += scores.mean()

scores = cross_val_score(NaiveBayes, x, y, cv=5)

print("NaiveBayes " + str(scores.mean()))

sumscores += scores.mean()

scores = cross_val_score(RandomForest, x, y, cv=5)

print("RandomForest " + str(scores.mean()))

sumscores += scores.mean()

scores = cross_val_score(ExtreemRandom, x, y, cv=5)

print("ExtreemRandom " + str(scores.mean()))

sumscores += scores.mean()

scores = cross_val_score(AdaBoost, x, y, cv=5)

print("AdaBoost " + str(scores.mean()))

sumscores += scores.mean()

scores = cross_val_score(LDA, x, y, cv=5)

print("LDA " + str(scores.mean()))

sumscores += scores.mean()

scores = cross_val_score(MLPClassifier, x, y, cv=5)

print("MLPClassifier " + str(scores.mean()))

sumscores += scores.mean()

scores = cross_val_score(QDA, x, y, cv=5)

print("QDA " + str(scores.mean()))

sumscores += scores.mean()


scores = cross_val_score(nuSVC, x, y, cv=5)

print("nuSVC " + str(scores.mean()))

sumscores += scores.mean()


scores = cross_val_score(GradientBoostingClassifier, x, y, cv=5)

print("GradientBoostingClassifier " + str(scores.mean()))

sumscores += scores.mean()