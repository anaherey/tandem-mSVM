# File: tandem_classification.py
# ------------------------------------------------------------------
# This file does the following functions:
#      1. Trains each SVM (SVM1,SVM2,SVM3)
#      2. Integrates all SVM into a tandem mSVM
#      3. Calculates the final error of the classifier
# ------------------------------------------------------------------
# Thesis: Multi-class Support Vector Machines for classification of 
#         brain hemodynamic patterns.
# Author: Ana G. Hernandez Reynoso
# Advisor: Alejandro Garcia Gonzalez
# Institution: Tecnologico de Monterrey

import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

# ------------------------------------------------------------------
# Training of SVM
# ------------------------------------------------------------------
# This function trains the SVM using Stratified K-Fold Cross
# Validation
# SVM = svm_train(X_train,y_train,k)
# - X_train: data for training the SVM
# - y_train: targets of the training data
# - k: number of folds
# ------------------------------------------------------------------
# Example:
# SVM1 = tandem_classification.svm_train(X1_train,y1_train,k)

def svm_train(X,y,k):
	C_range = 10.0 ** np.arange(-2, 9)
	gamma_range = 10.0 ** np.arange(-5, 4)
	param_grid = dict(gamma=gamma_range, C=C_range)
	cv = StratifiedKFold(y=y,n_folds=k)
	svm = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
	svm.fit(X,y)
	return svm

# ------------------------------------------------------------------
# Tandem mSVM 
# ------------------------------------------------------------------
# Executes the final classification using the tandem structure
# defined in the classification layer.
#       - If SVM1 classifies as 0 it goes to SVM2, otherwise it goes
#        to SVM3.
#       - If SVM2 classifies as 0 then is a non induced lie,
#        otherwise it is a non induced truth
#       - If SVM3 classifies as 2 then is an induced lie, otherwise
#        it is an induced truth
# mSVM = msvm_class(X,SVM1,SVM2,SVM3)
# - X: input Vector
# - SVM1: Trained SVM that classifies non induced from induced
# - SVM2: Trained SVM that classifies the subclasses lie/truth for 
#         non induced
# - SVM3: Trained SVM that classifies the subclasses lie/truth for
#         induced
# ------------------------------------------------------------------
# Example:
# Output = tandem_classification.msvm_class(X,SVM1,SVM2,SVM3)

def msvm_class(X,svm1,svm2,svm3):
	results_tandem = []
	for vector in range(0,len(X)):
		if svm1.predict(X[vector]) == 0:
			if svm2.predict(X[vector]) == 0:
				results_tandem.append(0)
			else:
				results_tandem.append(1)
		else:
			if svm3.predict(X[vector]) == 2:
				results_tandem.append(2)
			else:
				results_tandem.append(3)
	return results_tandem

# ------------------------------------------------------------------
# Error calculation
# ------------------------------------------------------------------
# Calculates the final error for the mSVM
# error = error_calc(Output,y)
# - Output: classifier output
# - y: target vector
# ------------------------------------------------------------------
# Example:
# error = tandem_classification.error_calc(Output,y)

def error_calc(Output,y):
	EmSVM = 0.
	for vector in range(0,len(Output)):
		if Output[vector] != y[vector]:
			EmSVM += 1
	EmSVM /= len(Output)
	return EmSVM*100

