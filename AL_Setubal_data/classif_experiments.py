# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:48:08 2020

@author: arman
"""
import os
import time

import numpy as np 
import pandas as pd 
from statistics import mean 
from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier

from statistics import stdev

#==========================================
# data preparation
#==========================================

# fixt the start time
t0 = time.time()

PATH = r'C:\Users\arman\Desktop\ActiveLearning\Experiment\dgt_agri'
data_path = os.path.join(PATH, 'Agri_data')

train = pd.read_csv(os.path.join(data_path,'training.csv'))
test = pd.read_csv(os.path.join(data_path,'testing.csv'))

dataset_soze = train.groupby('CLASS').size()
print('sample size:', train.shape[0], '\n'+'features:', train.shape[1])
print('dataset_soze: ', dataset_soze)
#print('dataset skewness', train.iloc[:, 2:].skew)

# define dependant and independant variables
X_train = train.iloc[:, 2:].values
y_train = train.iloc[:, 1].values

X_test = test.iloc[:, 2:].values
y_test = test.iloc[:, 1].values

# Normalize the training data
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)


#==========================================
# classification
#==========================================

# create lists for accuracies
ac1, ac2, ac3, ac4 = [], [], [], []

print('Training SVM')
'''SVC model '''
classifier1 = SVC(C=1, kernel='rbf') 
classifier1.fit(X_train, y_train) 
ac1.append(classifier1.score(X_test, y_test)) 
# time for classification wit SVM
t1 = time.time()

print('Training RF')
'''RF model'''
classifier2 = RandomForestClassifier(n_estimators=300) 
classifier2.fit(X_train, y_train) 
ac2.append(classifier2.score(X_test, y_test)) 
# time for classification wit FF
t2 = time.time()

print('Training LR')
'''LR model'''
param = {
        'C': 50,
        'multi_class': 'multinomial',
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 1200,
        }
classifier3 = LogisticRegression(**param) 
classifier3.fit(X_train, y_train) 
ac3.append(classifier3.score(X_test, y_test)) 
# time for classification wit FF
t3 = time.time()

'''GBC model'''
#classifier4 = GradientBoostingClassifier() 
#classifier4.fit(X_train, y_train) 
#ac4.append(classifier4.score(X_test, y_test))


print("Accuracy by SVM :", ac1[0]*100, 'model training time:', round(t1-t0), 'sec')
print("Accuracy by RF :", ac2[0]*100,  'model training time:', round(t2-t1), 'sec')
print("Accuracy by LG :", ac3[0]*100,  'model training time:', round(t3-t2), 'sec')
#print("Accuracy by GBC :", mean(ac4)*100, "stdev: ", stdev(ac4)*100)

# simple classification accuracies are
SVC = 0.7429845626072041, 
RF = 0.7552658662092624, 
LR = 0.7278902229845626, 


