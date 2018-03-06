###############################################
## Written by Marlena Duda for GuanLab
## USE: train various classifiers 
##      using cross validation to estimate AUC
## INPUT: brain_matrix_training.txt
## OUTPUT: print cross validation AUC
###############################################

import numpy as np
from sklearn import svm, grid_search
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

## Load Data ##
matrix = np.loadtxt("brain_matrix_training.txt", delimiter = "\t", skiprows = 1, usecols= xrange(1,21123))
target = np.genfromtxt("brain_matrix_training.txt", delimiter = "\t", skip_header = 1, usecols = -1)

## SVC trial ##
skf = StratifiedKFold(target, n_folds = 5)
for train, test in skf:
    parameters = {"C":[0.01,0.1,1,10,100]}
    est = svm.SVC(kernel = "linear",probability = True,class_weight = "balanced")
    clf = grid_search.GridSearchCV(est, parameters, n_jobs = 10)
    clf.fit(matrix[train], target[train])
    yPred = clf.predict_proba(matrix[test])[:,1]
    print roc_auc_score(target[test], yPred), "SVC"

## Random Forest ##
skf = StratifiedKFold(target, n_folds = 5)
for train, test in skf:
    clf = RandomForestClassifier(n_estimators = 200, class_weight = "balanced_subsample", n_jobs = -1, bootstrap = True)
    clf.fit(matrix[train], target[train])
    yPred = clf.predict_proba(matrix[test])[:,1]
    print roc_auc_score(target[test], yPred), "RandomForest"

## AdaBoost ##
skf = StratifiedKFold(target, n_folds = 5)
for train, test in skf:
    clf = AdaBoostClassifier(base_estimator = RandomForestClassifier(class_weight = "balanced_subsample"), n_estimators = 250)
    clf.fit(matrix[train], target[train])
    yPred = clf.predict_proba(matrix[test])[:,1]
    print roc_auc_score(target[test], yPred), "AdaBoostRandomForest"

## Boosting ##
skf = StratifiedKFold(target, n_folds = 5)
for train, test in skf:
    clf = BaggingClassifier(base_estimator = RandomForestClassifier(class_weight = "balanced_subsample"), n_estimators = 250, bootstrap = True, bootstrap_features = True, n_jobs = -1)
    clf.fit(matrix[train], target[train])
    yPred = clf.predict_proba(matrix[test])[:,1]
    print roc_auc_score(target[test], yPred), "BaggingRandomForest"

## Voting ##
skf = StratifiedKFold(target, n_folds = 5)
for train, test in skf:
    clf1 = RandomForestClassifier(class_weight = "balanced_subsample", n_jobs = -1)
    clf2 = svm.SVC(kernel = "linear", class_weight = "balanced", probability = True, C = 10)
    vclf = VotingClassifier(estimators = [('rf',clf1),('svc', clf2)], voting = "soft")
    vclf.fit(matrix[train], target[train])
    yPred = vclf.predict(matrix[test])
    print roc_auc_score(target[test], yPred)

## Extra Trees ##
skf = StratifiedKFold(target, n_folds = 5)
for train, test in skf:
    clf = ExtraTreesClassifier(n_estimators = 100, class_weight = "balanced_subsample", n_jobs = -1, bootstrap = True)
    clf.fit(matrix[train], target[train])
    yPred = clf.predict_proba(matrix[test])[:,1]
    print roc_auc_score(target[test], yPred), "ERTrees"

## Gradient Boost ##
skf = StratifiedKFold(target, n_folds = 5)
for train, test in skf:
    clf = GradientBoostingClassifier(n_estimators = 250, max_features = "auto", init = RandomForestClassifier(class_weight = "balanced_subsample", n_jobs = -1))
    clf.fit(matrix[train], target[train])
    yPred = clf.predict_proba(matrix[test])[:,1]
    print roc_auc_score(target[test], yPred), "GradientBoostRandomForest"

