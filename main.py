# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import FeatureSelection

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.linear_model import LinearRegression
from matplotlib.lines import Line2D
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score


#%% declare variables
# X for temperature, Y for thermal error
Xtrain, Xtest, Xpred=[], [], []
Ytrain, Ytest, Ypred=[], [], []

X={"Xtrain":Xtrain, "Xtest":Xtest, "Xpred":Xpred}
Y={"Ytrain":Ytrain, "Ytest":Ytest, "Ypred":Ypred}


#%% load data

for i in range(11):
    # load rawdata
    Xraw = pd.read_csv('train/traindata{0}.csv'.format(i+1), index_col=False, header=None).iloc[:,:-1].values
    Yraw = pd.read_csv('train/traindata{0}.csv'.format(i+1), index_col=False, header=None).iloc[:,-1].values.reshape(-1, 1)
    
    # split train dataset into train & test
    X_train, X_test, y_train, y_test = train_test_split(Xraw, Yraw, shuffle=False, test_size=0.33)
    
    # load predict dataset
    X_pred = pd.read_csv('test/traindata{0}.csv'.format(i+1), index_col=False, header=None).iloc[:,:-1].values
    y_pred = pd.read_csv('test/traindata{0}.csv'.format(i+1), index_col=False, header=None).iloc[:,-1].values.reshape(-1, 1)
    
    # load into dictionary
    Xtrain.append(X_train), Xtest.append(X_test), Xpred.append(X_pred)
    Ytrain.append(y_train), Ytest.append(y_test), Ypred.append(y_pred)


#%% optimize

trainingPara={
    "agent":20,
    "ndim":Xtrain[0].shape[1],
    "pos":len(Xtrain), 
    "iter_max":10
    }

# opt=FeatureSelection.FeatureSelection(agent=30, ndim=Xtrain[0].shape[1], pos=len(Xtrain), _iter_max= 10)
opt=FeatureSelection.FeatureSelection(trainingPara)

opt.Optimize(Xtrain, Ytrain)

opt.Predict(Xpred, Ypred)



#%% check result
a,b=opt.OutputPara()




