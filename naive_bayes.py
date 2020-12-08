from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from pandas import read_csv
from sklearn import preprocessing
import numpy as np
import math

pd.options.mode.chained_assignment = None  # default='warn'

names = ['quarter', 'down', 'yardsToGo', 'yardlineNumber',
# 'offenseFormation',
# 'personnelO',
# 'defendersInTheBox',
'numberOfPassRushers',
# 'personnelD',
# 'typeDropback',
'preSnapVisitorScore', 'preSnapHomeScore',
#'gameClock',
'absoluteYardlineNumber', 'passResult']

dataNames = ['quarter', 'down', 'yardsToGo', 'yardlineNumber',
# 'offenseFormation',
# 'personnelO',
# 'defendersInTheBox',
'numberOfPassRushers',
# 'personnelD',
# 'typeDropback',
'preSnapVisitorScore', 'preSnapHomeScore',
#'gameClock',
'absoluteYardlineNumber']

resultNames = ['passResult']

fullDataset = read_csv('Data/plays.csv')

features = fullDataset[names]

allClear = features.dropna()

finalData = allClear[names]

scaler = preprocessing.StandardScaler()

X = finalData[dataNames]
Y = finalData[resultNames]

X_Scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_Scaled, Y, test_size=0.5, random_state=0)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train.values.ravel()).predict(X_test)
print(y_pred[0])

y_pred = y_pred.reshape(len(y_pred),1)

print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))