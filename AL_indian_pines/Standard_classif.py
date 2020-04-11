# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 21:23:27 2020

@author: arman
"""
import os
import time

import numpy as np 
import pandas as pd 
import pandas_profiling
from statistics import mean, stdev
from collections import Counter
import pickle
import itertools
from sklearn.preprocessing import StandardScaler 
from sklearn.datasets import fetch_openml
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split 

# Download the dataset
def download():
    # Doanload Indian Pines dataset from OpenML platform
    X, y = tuple(fetch_openml('Indian_pines').values())[:2]
        
    # Unmute the line to binerize the data 
    #y = np.array((y== 'Soybeans').astype(int)) 

    print ('Indianpines dataset:', '\nN samples:',X.shape[0],'\nN features:',X.shape[1],'\n')
    print (f'Classes size:\n{pd.Series(dict(Counter(y).most_common()))}')
    dataset =  np.column_stack((y, X.astype(np.object)))
    return (X, y)

# Load and save pickles
def pickle_save(fname, data):
    filehandler = open(fname,"wb")
    pickle.dump(data,filehandler)
    filehandler.close() 
    print('saved', fname, os.getcwd(), os.listdir())

def pickle_load(fname):
    file = open(fname,'rb')
    data = pickle.load(file)
    file.close()
    return data

X,y = download()

# For a comprehensive dataset description unmute and run the shell
#pandas_profiling.ProfileReport(pd.DataFrame(pd.DataFrame(np.column_stack((y, X.astype(np.object))),
#                                        columns=['label']+[str(i) for i in range(X.shape[0])])))


# Define list of accuracies for each classifier
acc1, acc2, acc3, acc4 = [], [], [], []

for _ in range(5):
    # Split the data into train/test
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=None, 
                                                            test_size=0.33, shuffle=True)
    
    # Normalize the training data
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    # Set up Random Forest classifier
    classifier1 = RandomForestClassifier(n_estimators=1000) 
    classifier1.fit(X_train, Y_train) 
    acc1.append(classifier1.score(X_test, Y_test)) 
    
    # Set up Logistic Regression classifier
    param2 = {
            'C': 50,
            'multi_class': 'multinomial',
            'penalty': 'l1',
            'solver': 'saga',
            'max_iter': 2500,
            }
    classifier2 = LogisticRegression(**param2) 
    classifier2.fit(X_train, Y_train) 
    acc2.append(classifier2.score(X_test, Y_test)) 
    
    # Set up Support Vector classifier
    param3 = {
            'C': 1,
            'kernel': 'linear',
            'gamma': 'auto'
            }
    classifier3 = SVC(**param3)
    classifier3.fit(X_train, Y_train) 
    acc3.append(classifier3.score(X_test, Y_test))
    
    '''
    # Set up Gradient Boosting classifier
    param4 = {
            'n_estimators': 1200,
            'max_depth': 3,
            'subsample': 0.5,
            'learning_rate': 0.01,
            'min_samples_leaf': 1,
            'random_state': 3,
            }
    classifier4 = GradientBoostingClassifier(**param4) 
    classifier4.fit(X_train, y_train) 
    acc4.append(classifier4.score(X_test, y_test))
    '''
print("Accuracy by RF :", mean(acc1)*100, "stdev: ", stdev(acc1)*100)
print("Accuracy by LogReg :", mean(acc2)*100, "stdev: ", stdev(acc2)*100)
print("Accuracy by SVM :", mean(acc3)*100, "stdev: ", stdev(acc3)*100)
#print("Accuracy by GBC :", mean(ac4)*100, "stdev: ", stdev(ac4)*100)

# Steore the results in a dictionary
standard = {}
standard['standard_RF'] = mean(acc1).round(2)
standard['standard_LogReg'] = mean(acc2).round(2)
standard['standard_SVM'] = mean(acc3).round(2)

print(standard)