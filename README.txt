tandem-mSVM v0.1 11/28/2014
=======================================================================

This is a tandem structure of binary Support Vector Machines to solve a multi-class problem that is composed of subclasses of classes. The structure is described in:

Thesis: Multi-class Support Vector Machines for classification of brain hemodynamic patterns. 
Author: Ana G. Hernandez-Reynoso
        anaherey@gmail.com
Advisor: Alejandro Garcia-Gonzalez 
Institution: Tecnologico de Monterrey


=======================================================================

LICENSE
---------------------------------
tandem-mSVM is a software for multi-class classification with an array of multiple Support Vector Machines in tandem for problems that have subclasses dependent on main classes.

Copyright (C) 2014 Ana HeRey

tandem-mSVM is free software; you can redistribute it and/or modify it under the terms of the GNU General Public Licence as published by the Free Software Foundationm, either version 3 of the Licence, or (at your option) any later version.

tandem-mSVM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public Licence for more details.

You should have received a copy of the GNU General Public Licence along with this program; if not, see <http://www.gnu.org/licenses/>.


=======================================================================

GENERAL USAGE NOTES
---------------------------------
- Program was develped under the Anaconda 2.0.1 Python Distribution  with Python 2.7.7 on darwin <https://store.continuum.io/cshop/anaconda/>.

- Program was developed for Macintosh OS X 10.9.5.

- This program solves a multi-class classification problem where it can be divided into subclasses dependent on main classes. This is a tandem (sequential) structure of multiple binary support vector machines, where classification is performed in stages.


=======================================================================

INSTALLATION
---------------------------------
1. Install Anaconda 2.0.1 from https://store.continuum.io/cshop/anaconda/ following website instructions.

2. Clone Git Repository from https://github.com/anaherey/tandem-mSVM to custom location.


=======================================================================

INSTRUCTIONS
---------------------------------
1. Open Terminal

2. Change working directory to repository directory (where repository was cloned).

3. Run using Anaconda Python 2.7.7
	Command Line:
		-$ python main_execution.py id
		-------------------------------
		Input:
			- id: subject id. It contains all the input-output pairs for classification of patterns for subject 'id'. 'id' must be a number. To run example use id == 0
		Outputs:
			- Training time: time to train classifier.
			- Execution time: time to execute classifier.
			- Classifier error: percentage of patterns classified incorrectly.
			- Output csv: csv file containing the output for each pattern.
		-------------------------------
		Example:
			-$ py27 main_execution.py 0


=======================================================================

CONTACT
---------------------------------

For more information about the tandem multi-class classifier with Support Vector Machines contact Author:
Author: Ana G. Hernandez-Reynoso
        anaherey@gmail.com
