# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 23:07:02 2019

@author: Toprak
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

LogisticRegression = LogisticRegression(multi_class='multinomial',solver='lbfgs')
KNeighbors = KNeighborsClassifier(n_neighbors = 100)
SupportVector = svm.SVC(gamma='scale',probability=True)
NaiveBayes = GaussianNB()
DecisionTree = tree.DecisionTreeClassifier()
RandomForest = RandomForestClassifier(n_estimators=10)
ExtreemRandom = ExtraTreesClassifier(n_estimators=10)
AdaBoost = AdaBoostClassifier()
LDA = LinearDiscriminantAnalysis()

MLPClassifier = MLPClassifier(alpha=1, max_iter=1000)
QDA = QuadraticDiscriminantAnalysis()

nuSVC = NuSVC(gamma='scale',probability=True)
GradientBoostingClassifier = GradientBoostingClassifier()


LogisticRegression.fit(x,y)
KNeighbors.fit(x,y)
SupportVector.fit(x,y)
NaiveBayes.fit(x,y)
DecisionTree.fit(x,y)
RandomForest.fit(x,y)
ExtreemRandom.fit(x,y)
AdaBoost.fit(x,y)
LDA.fit(x,y)
MLPClassifier.fit(x,y)
QDA.fit(x,y)

nuSVC.fit(x,y)
GradientBoostingClassifier.fit(x,y)