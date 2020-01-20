# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:09:20 2019

@author: arman
"""
import os
import time

import numpy as np 
import pandas as pd 
from statistics import mean 
from collections import Counter
import pickle
from sklearn.preprocessing import StandardScaler 
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split 
from statistics import stdev

#==============================================================================
''' Indian pines dataset '''
def download():
    X, y = tuple(fetch_openml('Indian_pines').values())[:2]
    #X = X.astype('float64')
    #y = np.array((y== 'Soybeans').astype(int)) 
    #y = np.array(['Soybeans' if i=='Soybeans' else 'Other' for i in y])
    class_size = Counter(y)
    classes = len(np.unique(y))
    print ('Indian_pines:', X.shape, y.shape, classes, class_size)
    dataset =  np.column_stack((y,X.astype(np.object)))
    return dataset

#dataset = download()
#==============================================================================
''' CLC data'''
def download2():
    path = r'C:\Users\arman\Desktop\New paper\CLC_data\dataset_10%.csv'
    dataset = pd.read_csv(path).values
    return dataset

#============================================================================== 
''' Split dataset '''
def split(dataset, train_size, test_size): 
    ''' for indian_pines '''
    x = dataset[:, 1:] 
    y = dataset[:, 0] 
    
    '''for CLC data'''
    #x = dataset[:, 3:] 
    #y = dataset[:, 0] 
                
    x_train, x_pool, y_train, y_pool = train_test_split( x, y, 
                        train_size = train_size, stratify = y) 
    
    unlabel, x_test, label, y_test = train_test_split( x_pool, y_pool, 
                    test_size = test_size, stratify = y_pool)
    
    # feature scalling 
    sc = StandardScaler() 
    
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    unlabel = sc.fit_transform(unlabel) 
    
    return x_train, y_train, x_test, y_test, unlabel, label 

#==============================================================================
    ''' DATA SELECTION STRATEGIES
    
    rank the prediction uncertainty for each predicted sample based on the 
    classification probability autput  '''

''' Uncertainty with Entropy value'''
def EntropySelection(y_probab):
    # entropy measure
    # avoid having absolut 0 for probab. values
    if model.__name__ =='RF':
        y_prob = np.where(np.isin(y_probab, [0,1]), np.clip(y_probab,1e-10,1), y_probab)
        e = (-y_prob * np.log2(y_prob)).sum(axis=1) 
    else:
        e = (-y_probab * np.log2(y_probab)).sum(axis=1)
    selection = (np.argsort(e)[::-1])[:step]
    selection = list(selection)
    return selection

''' Uncertainty with Margine sampling '''
def MarginSamplingSelection(y_probab):
    # Selecting Margine samples
    rev = np.sort(y_probab, axis=1)[:, ::-1]
    values = rev[:, 0] - rev[:, 1]
    selection = np.argsort(values)[:step]
    return selection

'''Random Selection'''
def RandomSelection(y_probab):
    selection = np.random.choice(y_probab.shape[0], step, replace=False)
    #selection = list(selection)
    return selection

#=======================CLASSIFICATION MODELS==================================
'''
Random Forest Classifier
'''
def RF():
    classifier = RandomForestClassifier(n_estimators=200)
    classifier.fit(x_train, y_train)
    y_probab = classifier.predict_proba(unlabel)
    score = classifier.score(x_test, y_test)
    return (y_probab, score)
   
'''
Logistic Regression
'''
def LogReg():
    param = {
            'C': 50,
            'multi_class': 'multinomial',
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1200,
            }
    classifier = LogisticRegression(**param)
    classifier.fit(x_train, y_train)
    y_probab = classifier.predict_proba(unlabel)
    score = classifier.score(x_test, y_test)
    return (y_probab, score)

'''
GradientBoostingClassifier
'''
def GBC():
    param = {
            'n_estimators': 1200,
            'max_depth': 3,
            'subsample': 0.5,
            'learning_rate': 0.01,
            'min_samples_leaf': 1,
            'random_state': 3,
            }
    classifier = GradientBoostingClassifier(**param)
    classifier.fit(x_train,y_train)
    y_probab = classifier.predict_proba(unlabel)
    score = classifier.score(x_test, y_test)
    return (y_probab, score)

'''
Support Vector Machine
'''
def SVM():
    param = {
            'C': 1,
            'kernel': 'rbf',
            'probability': True,
            'gamma': 'auto'
            }
    classifier = SVC(**param)
    classifier.fit(x_train,y_train)
    y_probab = classifier.predict_proba(unlabel)
    score = classifier.score(x_test, y_test)
    return (y_probab, score)

#==============================================================================
def pickle_save(fname, data):
  filehandler = open(fname,"wb")
  pickle.dump(data,filehandler)
  filehandler.close() 
  print('saved', fname, os.getcwd(), os.listdir())

def pickle_load(fname):
  #print(os.getcwd(), os.listdir())
  file = open(fname,'rb')
  data = pickle.load(file)
  file.close()
  #print(data)
  return data

#==============================================================================
'''EXPERIMENT SETUP'''
steps = [10, 20, 40]
iters = 5 # number of iterations within each selection step
models = [RF, SVM, LogReg]#, GBC]
selection_functions = [MarginSamplingSelection, EntropySelection, RandomSelection]

#===================================EXPERIMENT=================================
if __name__ == '__main__':
    
    #Loading the data
    dataset = download()
    
    # store accuracy of different combinations 
    #acc = {}
    
    for step in steps:# count of selected sample in each iteration
        
        acc = {} # store accuracy of different combinations 
               
        for model in models:   # classifier
            print(model.__name__)

            for selection_function in selection_functions:  # sample selection strategy
                print(model.__name__+ '_' +selection_function.__name__ +'_'+ str(step))
                acc[model.__name__+ '_' +selection_function.__name__ +'_'+ str(step)] = {}
                                
                for i in range(1,iters+1): # number of iterations
                    print('Iter: ', i)
            
                    # split dataset into train(5 %), test(25 %), unlabel(70 %) 
                    x_train, y_train, x_test, y_test, unlabel, label = split(dataset, 0.02, 0.25) 
    
                    # let the active method select up to ~20% of training data
                    max_training = round(0.25*len(label)) 
                    #max_training = len(label)-20
                    
                    # create an empty list inside the dictionary for each iteration
                    acc[model.__name__+ '_' +selection_function.__name__+'_'+ str(step)][f'iter_{i}'] = []
                    
                    # record the training data for each iteration
                    acc[f'step_{step}'] = []
    
                    '''Active Selection'''
                    while len(x_train) <= max_training:
                        # probability by a classifier 
                        y_probab = model()[0]
                        # uncertainty points by a given selection strategy
                        uncrt_pt_ind = selection_function(y_probab)
    
                        #print('x_train before Iter', x_train.shape[0])
                        x_train = np.append(unlabel[uncrt_pt_ind, :], x_train, axis = 0) 
                        y_train = np.append(label[uncrt_pt_ind], y_train) 
                        unlabel = np.delete(unlabel, uncrt_pt_ind, axis = 0) 
                        label = np.delete(label, uncrt_pt_ind) 
                            
                        # record the accuracy after every iteration
                        #score = model(x_train, y_train, x_test, y_test, unlabel)[1]
                        score = model()[1]
                        #print('x_train after Iter', x_train.shape[0])
                        #acc[model.__name__+ '_' +selection_function.__name__+'_'+ str(step)].append(score)
                        
                        acc[model.__name__+ '_' +selection_function.__name__+'_'+ str(step)][f'iter_{i}'].append(score)
                        
                        # record the training data after each iteration
                        acc[f'step_{step}'].append(len(x_train))
            
            # save pickled disctionary with results
            fname = 'AL_full_run_CLC_' + str(step) + '.pkl'
            pickle_save(fname, acc)

#==============================================================================
# Load the pickled file anc check the results
#step = 20
#acc = pickle_load('AL_full_experiment_Indian_pine_' + str(step)+'.pkl')

#acc.keys()









    
    
    
