# File: data_controller.py
# ------------------------------------------------------------------
# This file does the following functions:
#      1. Read the CSV file that contains the examples vectors
#      2. Read the CSV file that contains the targets vector
#      3. Separates the data into training and testing subsets
#      4. Saves the final output into a CSV file
# ------------------------------------------------------------------
# Thesis: Multi-class Support Vector Machines for classification of 
#         brain hemodynamic patterns.
# Author: Ana G. Hernandez Reynoso
#         anaherey@gmail.com
# Advisor: Alejandro Garcia Gonzalez
# Institution: Tecnologico de Monterrey

"""
tandem-mSVM is a software for multi-class classification with
  an array of multiple Support Vector Machines in tandem for 
  problems that have subclasses dependent on main classes.
Copyright (C) 2014 Ana HeRey

This file 'main_execution.py' is part of tandem-mSVM.

tandem-mSVM is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public Licence
  as published by the Free Software Foundationm, either version
  3 of the Licence, or (at your option) any later version.

tandem-mSVM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
  GNU General Public Licence for more details.

You should have received a copy of the GNU General Public Licence
  along with this program; if not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from sklearn.cross_validation import KFold
import os

# ------------------------------------------------------------------
# Read input vector
# ------------------------------------------------------------------
# This function reads the input vectors from a CSV file
# X = read_input(subject)
# - subject: subject ID number
# ------------------------------------------------------------------
# Example:
# X = data_controller.read_input(1)

def read_input(subject):
	pwd = os.getcwd()
	data_dir = os.path.join(pwd+"/data/subject"+str(subject)+"_data.csv")
	if subject == 0: # If example is executed
		data_dir = os.path.join(pwd+"/example/data/example_data.csv")
	data = np.loadtxt(open(data_dir,"rb"),delimiter=",",
		skiprows=1)
	return data

# ------------------------------------------------------------------
# Read target vector
# ------------------------------------------------------------------
# This function reads the target vectors from a CSV file
# X = read_target(subject)
# - subject: subject ID number
# ------------------------------------------------------------------
# Example:
# X = data_controller.read_target(1)

def read_target(subject):
	pwd = os.getcwd()
	data_dir = os.path.join(pwd+"/data/subject"+str(subject)+"_target.csv")
	if subject == 0: # If example is executed
		data_dir = os.path.join(pwd+"/example/data/example_target.csv")
	data = np.loadtxt(open(data_dir,"rb"),delimiter=",",
		skiprows=1)
	return data

# ------------------------------------------------------------------
# Select pairs for SVM1
# ------------------------------------------------------------------
# [X1,y1] = pairs_svm1(X,y)
# - X: original input set
# - y: original target set
# ------------------------------------------------------------------
# Example:
# [X1,y1] = data_controller.pairs_svm1(X,y)

def pairs_svm1(X,y):
	y1 = []
	for i in range(0,len(y)):
		if (y[i] == 0) or (y[i] == 1):
			y1.append(0)
		if (y[i] == 2) or (y[i] == 3):
			y1.append(1)
	return (X,np.asarray(y1))

# ------------------------------------------------------------------
# Select pairs for SVM2
# ------------------------------------------------------------------
# [X2,y2] = pairs_svm2(X,y)
# - X: original input set
# - y: original target set
# ------------------------------------------------------------------
# Example:
# [X2,y2] = data_controller.pairs_svm2(X,y)

def pairs_svm2(X,y):
	y2 = []
	X2 = []
	for i in range(0,len(y)):
		if (y[i] == 0) or (y[i] == 1):
			y2.append(y[i])
			X2.append(X[i])
	return (np.asarray(X2),np.asarray(y2))

# ------------------------------------------------------------------
# Select pairs for SVM3
# ------------------------------------------------------------------
# [X3,y3] = pairs_svm3(X,y)
# - X: original input set
# - y: original target set
# ------------------------------------------------------------------
# Example:
# [X3,y3] = data_controller.pairs_svm3(X,y)

def pairs_svm3(X,y):
	y3 = []
	X3 = []
	for i in range(0,len(y)):
		if (y[i] == 2) or (y[i] == 3):
			y3.append(y[i])
			X3.append(X[i])
	return (np.asarray(X3),np.asarray(y3))

# ------------------------------------------------------------------
# Uses cross-validation for selecting training/testing subsets
# ------------------------------------------------------------------
# [X_train,X_test,y_train,y_test] = cvkfold(X,y,k,)
# - X: Input set
# - y: Target set
# - k: Number of folds
# ------------------------------------------------------------------
# Example:
# [X1_train,X1_test,y1_train,y2_test] = data_controller.cvkfold(X1,y1,k)

def cvkfold(X,y,k,ran):
	skf = KFold(n=len(X),n_folds=k,shuffle=True,random_state=ran)
	# random_state=True selects the most representative indexes
	for train_index, test_index in skf:
		X_train, X_test = X[train_index],X[test_index]
		y_train, y_test = y[train_index],y[test_index]
	return (X_train,X_test,y_train,y_test)

# ------------------------------------------------------------------
# Saves data to CSV file
# ------------------------------------------------------------------
# save_data(data,index)
# - data: data to be saved
# - subject: subject ID number
# ------------------------------------------------------------------
# Example:
# data_controller.save_data(Output,1)

def save_data(data,subject):
	pwd = os.getcwd()
	out_dir = os.path.join(pwd+"/results/subject"+str(subject)+"_output.csv")
	if subject == 0:
		out_dir = os.path.join(pwd+"/example/results/example_output.csv")
	np.savetxt(out_dir,data,delimiter=',',fmt="%.0f")
	return 0