from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB

from sklearn.cluster import KMeans

from sklearn import preprocessing

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

from pandas import read_csv

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import math
import sys

pd.options.mode.chained_assignment = None  # default='warn'

names = [
'quarter',
'down',
'yardsToGo',
'yardlineNumber',
'numberOfPassRushers',
'preSnapVisitorScore',
'preSnapHomeScore',
'absoluteYardlineNumber',
'epa',
# 'defendersInTheBox',
'typeDropback',
'gameClock',
'offenseFormation',
'numDL',
'numLB',
'numDB',
'numRB',
'numTE',
'numWR',
# 'isDefensivePI'
# 'penaltyCodes'
# 'offensePlayResult'
'passResult'
]

dataNames = [
'quarter',
'down',
'yardsToGo',
'yardlineNumber',
'numberOfPassRushers',
'preSnapVisitorScore',
'preSnapHomeScore',
'absoluteYardlineNumber',
'gameClock',
'typeDropback',
'offenseFormation',
'numDL',
'numLB',
'numDB',
'numRB',
'numTE',
'numWR',
# 'defendersInTheBox',
'epa'
]
# resultNames = ['isDefensivePI']

# resultNames = ['penaltyCodes']

# resultNames = ['offensePlayResult']

resultNames = ['passResult']

fullDataset = read_csv('Data/plays.csv')

features = fullDataset[names]

# Get rid of any row that doesn't have one of the datapoints we're looking at
allClear = features.dropna()

finalData = allClear[names]

X = finalData[dataNames]
Y = finalData[resultNames]

# Make typeDropback pretty
X.typeDropback[X.typeDropback == 'TRADITIONAL'] = 0
X.typeDropback[X.typeDropback == 'SCRAMBLE_ROLLOUT_LEFT'] = 1
X.typeDropback[X.typeDropback == 'DESIGNED_ROLLOUT_LEFT'] = 2
X.typeDropback[X.typeDropback == 'SCRAMBLE_ROLLOUT_RIGHT'] = 3
X.typeDropback[X.typeDropback == 'DESIGNED_ROLLOUT_RIGHT'] = 4
X.typeDropback[X.typeDropback == 'SCRAMBLE'] = 5
X.typeDropback[X.typeDropback == 'UNKNOWN'] = 6

X.offenseFormation[X.offenseFormation == 'I_FORM'] = 0
X.offenseFormation[X.offenseFormation == 'SINGLEBACK'] = 1
X.offenseFormation[X.offenseFormation == 'SHOTGUN'] = 2
X.offenseFormation[X.offenseFormation == 'EMPTY'] = 3
X.offenseFormation[X.offenseFormation == 'PISTOL'] = 4
X.offenseFormation[X.offenseFormation == 'WILDCAT'] = 5
X.offenseFormation[X.offenseFormation == 'JUMBO'] = 6

if sys.argv[2] == 'lda':
	# Dimension Reduction via LDA (From Naive Bayes)
	scaler = LDA()
	X = scaler.fit_transform(X,Y.values.ravel())

elif sys.argv[2] == 'pca':
	# Dimension Reduction via PCA
	pca = PCA(n_components='mle')
	X = pca.fit_transform(X)

elif sys.argv[2] == 'norm':
	# Normal Scaler
	scaler = StandardScaler()
	X = scaler.fit_transform(X)

accuracy = []

if sys.argv[1] == 'bayes':
	# Naive Bayes Code
	clf = GaussianNB()

	accuracy = cross_val_score(clf, X,Y.values.ravel(), cv=5)

	print("Naive Bayes Accuracy: " + str(np.mean(accuracy)))

	accuracy = []
elif sys.argv[1] == 'svm':
	# SVM Code
	clf = SVC(gamma='auto')

	accuracy = cross_val_score(clf, X,Y.values.ravel(), cv=5)

	print("SVM Accuracy: " + str(np.mean(accuracy)))
elif sys.argv[1] == 'nnet':
	alphas = [.0001, .001, .01, .1, 1, 10]
	for i in alphas:
		accuracy = []
		clf = MLPClassifier(solver='lbfgs', alpha=i, hidden_layer_sizes=(20, 15), random_state=1, max_iter=10000)

		accuracy = cross_val_score(clf, X,Y.values.ravel(), cv=5)

		print("For Alpha: " + str(i))
		print("Total Accuracy: " + str(np.mean(accuracy)))