import numpy as np 
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

# Read attributes
with open('data/spam_classification_dense/DENSE.TRAIN.X.100', 'r') as f:
	data = []
	for line in f:
		data.append([int(x) for x in line.split(',')])
# Read labels
with open('data/spam_classification_dense/DENSE.TRAIN.Y.100', 'r') as f:
	label = []
	for line in f:
		label.append(int(line))

feature_num = 2

data_new = SelectKBest(chi2, k=feature_num).fit_transform(data, label)

