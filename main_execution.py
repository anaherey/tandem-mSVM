# File: main_execution.py
# ------------------------------------------------------------------
# This script is the main execution for the tandem mSVM for
#       classifying classes of subclasses
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
import data_controller
import tandem_classification
import sys
import os
import time

subject = int(sys.argv[1]) # Subject being analyzed

# ------------------------------------------------------------------
# GNU GPLv3
# ------------------------------------------------------------------
print("\n tandem-mSVM v0.1, Copyright (C) 2014 Ana HeRey \n")
option = raw_input("Type 'more' for more information about the\
 license.\n Press 'Return' to continue.\n")

if option.lower() == 'more':
  f = open('COPYING.txt')
  license = f.read()
  print(license)
  print('\n \n')
  f.close()
  #print("tandem-mSVM is a software for multi-class classification with an array of multiple Support Vector Machines in tandem for problems that have subclasses dependent on main classes.\n Copyright (C) 2014 Ana HeRey \n \n This file 'main_execution.py' is part of tandem-mSVM. \n \n tandem-mSVM is free software; you can redistribute it and/or modify it under the terms of the GNU General Public Licence as published by the Free Software Foundationm, either version 3 of the Licence, or (at your option) any later version.\n \n tandem-mSVM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public Licence for more details.\n \n You should have received a copy of the GNU General Public Licence along with this program; if not, see <http://www.gnu.org/licenses/>.")

# ------------------------------------------------------------------
# Read input/target pair Vectors from CSV files
# ------------------------------------------------------------------
# Input structure:
X = data_controller.read_input(subject)
""" Columns are variables:
          - 1:9 = Mean Deoxyhemoglobin left to right hemispheres.
          - 10:18 = Mean Oxyhemoglobin left to right hemispheres.
          - Voxels: [1,3,5,6,7,8,10,12,14].
    Rows are the examples.
"""

# Target structure:
y = data_controller.read_target(subject)
""" Rows are the targets for each example.
    Targets are [0,1,2,3] that represent:
         - 0 = Non induced lie
         - 1 = Non induced truth
         - 2 = Induced lie
         - 3 = Induced Truth
"""

# ------------------------------------------------------------------
# Select pairs for SVM1, SVM2 and SVM3
# ------------------------------------------------------------------
[X1,y1] = data_controller.pairs_svm1(X,y)
""" Selects all X as input for SVM1
    Transforms targets for having two classes:
          - 0 = Non induced respones (0 and 1)
          - 1 = Induced respones (2 and 3)
"""

[X2,y2] = data_controller.pairs_svm2(X,y)
""" Selects 0 and 1 from X as input for SVM2
    Transforms targets for having two classes:
          - 0 = Non induced lies
          - 1 = Non induced truths
"""

[X3,y3] = data_controller.pairs_svm3(X,y)
""" Selects 2 and 3 from X as input for SVM2
    Transforms targets for having two classes:
          - 2 = Induced lies
          - 3 = Induced truths
"""

# ------------------------------------------------------------------
# Selection of trainning/testing sets using Cross Validation
# ------------------------------------------------------------------
""" Selects the training and test subsets for the SVMs
    Uses K-Fold Cross Validation to select subsets
    k = number of folds
    Yields a test set of 15 percent of data
"""
# Cross Validation for SVM1 ----------------------------------------
k = 7 # Number of folds to yield 15 percent for test set
ran = True
if subject == 0:
	ran = 0
[X1_train,X1_test,y1_train,y1_test] = data_controller.cvkfold(X1,y1,k,ran)

# Cross Validation for SVM2 ----------------------------------------
k = 5 # Number of folds to yield 15 percent for test set
ran = True
[X2_train,X2_test,y2_train,y2_test] = data_controller.cvkfold(X2,y2,k,ran)

# Cross Validation for SVM3 ----------------------------------------
k = 6 # Number of folds to yield 15 percent for test set
ran = True
if subject == 0:
	ran = 0
[X3_train,X3_test,y3_train,y3_test] = data_controller.cvkfold(X3,y3,k,ran)


# ------------------------------------------------------------------
# SVM's trainning
# ------------------------------------------------------------------
start_time = time.time()
""" Trains the SVMs using Stratified K-Fold Cross Validation
    where k = number of folds """
# SVM1 -------------------------------------------------------------
k = 7 # Number of folds to yield 15 percent for test set
SVM1 = tandem_classification.svm_train(X1_train,y1_train,k)


# SVM2 -------------------------------------------------------------
k = 2 # Number of folds to yield 15 percent for test set
SVM2 = tandem_classification.svm_train(X2_train,y2_train,k)


# SVM3 -------------------------------------------------------------
k = 5 # Number of folds to yield 15 percent for test set
SVM3 = tandem_classification.svm_train(X3_train,y3_train,k)
train_time = time.time() - start_time
print('Training time: '+ str(train_time) + ' s')

# ------------------------------------------------------------------
# Tandem mSVM 
# ------------------------------------------------------------------
start_time = time.time()
""" Executes the final classification using the tandem structure
    defined in the classification layer in the following classes:
    - Non induced lie
    - Non induced truth
    - Induced lie
    - Induced truth
"""
Output = tandem_classification.msvm_class(X,SVM1,SVM2,SVM3)
exec_time = time.time() - start_time
print('Execution time: ' + str(exec_time*1000) + ' ms')

# ------------------------------------------------------------------
# Final error calculation
# ------------------------------------------------------------------
""" Calculates the final error in percentage for the tandem mSVM """
error = tandem_classification.error_calc(Output,y)
print("Tandem mSVM classification error: "+ str(error)+"%")

# ------------------------------------------------------------------
# Save data to csv file
# ------------------------------------------------------------------
data_controller.save_data(Output,subject)