# ecg-classification
ECG-based Arrhythmia Classification

# dataset
UCI Arrhythmia Dataset
http://archive.ics.uci.edu/ml/datasets/arrhythmia

The original data file is renamed from "arrhythmia.data" to "arrhythmia.data.orig" in this repository.

"arrhythmia.data" in this repository is the preprocessed file with all the question marks replaced by empty strings "".

"arrhythmia.names" is the codebook of the dataset.

# depends
	Python 3
	ipython (5.3.0)
	jupyter (1.0.0)
	matplotlib (1.4.2)
	notebook (4.4.1)
	numpy (1.12.1)
	pandas (0.19.2)
	scikit-learn (0.18.1)
	scipy (0.19.0)
	xgboost (0.6a2)

Python 3 should be installed first.
To install the above Python packages, run
	pip install <package_name>

# run and browse the Jupyter notebook
	$ jupyter notebook
	Open ecg-exp.ipynb in Jupyter.
