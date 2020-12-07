from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from pandas import read_csv
import numpy as np
import math

names = ['quarter', 'down', 'yardsToGo', 'yardlineNumber', 'offenseFormation',
# 'personnelO',
'defendersInTheBox', 'numberOfPassRushers',
# 'personnelD',
'typeDropback', 'preSnapVisitorScore', 'preSnapHomeScore',
#'gameClock',
'absoluteYardlineNumber']

fullDataset = read_csv('Data/plays.csv')

features = fullDataset[names]

# print(features.preSnapVisitorScore.notna())

for i in names:
	allClear = features[features[i].notna()]

for i in names:
	print(i)
	print(allClear[i].unique())
# print(test2[0,0])
# X, y = load_iris(return_X_y=True)
# print(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
# gnb = GaussianNB()
# y_pred = gnb.fit(X_train, y_train).predict(X_test)
# print("Number of mislabeled points out of a total %d points : %d"
#       % (X_test.shape[0], (y_test != y_pred).sum()))