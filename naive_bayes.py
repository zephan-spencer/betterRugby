from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB

from sklearn.cluster import KMeans

from sklearn import preprocessing

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

from pandas import read_csv

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import math

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
# 'personnelO',
# 'personnelD',
# 'gameClock',
# 'isDefensivePI'
# 'penaltyCodes'
'offensePlayResult'
# 'passResult'
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
# 'defendersInTheBox',
# 'personnelO',
# 'personnelD',
# 'gameClock',
'epa'
]


# resultNames = ['isDefensivePI']

# resultNames = ['penaltyCodes']

resultNames = ['offensePlayResult']

# resultNames = ['passResult']

orignalData = read_csv('Data/plays.csv')
wantedData = orignalData[names]
# Get rid of any row that doesn't have one of the datapoints we're looking at
features = wantedData.dropna()

X = features[dataNames]
Y = features[resultNames]

# Dimension Reduction via LDA
# scaler = LDA()
# X = scaler.fit_transform(X,Y.values.ravel())

# Dimension Reduction via PCA
pca = PCA(n_components='mle')
X = pca.fit_transform(X)

accuracy = []

for i in range(10,100):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

	# gnb = ComplementNB(norm = True)

	# Need to explore the multinomial classifier a bit more
	gnb = GaussianNB()
	y_pred = gnb.fit(X_train, y_train.values.ravel()).predict(X_test)

	# print("Class Priors: " + str(gnb.class_prior_[1]))

	y_pred = y_pred.reshape(len(y_pred),1)

	print("Number of mislabeled points out of a total %d points : %d"
	      % (X_test.shape[0], (y_test != y_pred).sum()))
	accuracy.append(1-((y_test != y_pred).sum())/X_test.shape[0])

print("Total Accuracy: " + str(np.mean(accuracy)))

# Y.passResult[Y.passResult == 'I'] = 0
# Y.passResult[Y.passResult == 'C'] = 1
# Y.passResult[Y.passResult == 'IN'] = 2
# Y.passResult[Y.passResult == 'S'] = 3
