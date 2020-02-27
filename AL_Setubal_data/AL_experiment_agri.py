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
# fixt the start time
t0 = time.time()

#==========================================
# data preparation
#==========================================
def download():
        
    PATH = r'C:\Users\mkhudinyan\Desktop\Arman\AL_experiments'
    data_path = os.path.join(PATH, 'Setubal_data')
    
    train = pd.read_csv(os.path.join(data_path,'training.csv'))
    test = pd.read_csv(os.path.join(data_path,'testing.csv'))
    
    train_dataset = train.groupby('CLASS').size()
    test_dataset = test.groupby('CLASS').size()
    print('sample size train:', train.shape[0], 
          '\nsample size test:', test.shape[0], 
          '\n'+'features:', train.shape[1])
    print('train dataset class\' size: ', train_dataset)
    print('test dataset class\' size: ', test_dataset)
    #print('dataset skewness', train.iloc[:, 2:].skew)
    
    return (train, test)


# Normalize the training data
#X_train = StandardScaler().fit_transform(X_train)
#X_test = StandardScaler().fit_transform(X_test)


#============================================================================== 
''' Split dataset '''
def split(train, test,  train_size): 
    ''' for indian_pines '''
    # define dependant and independant variables
    x = train.iloc[:, 2:].values
    y = train.iloc[:, 1].values
    
    x_test = test.iloc[:, 2:].values
    y_test = test.iloc[:, 1].values
    
    x_train, unlabel, y_train, label = train_test_split( x, y, 
                        train_size = train_size, stratify = y)    
    
    # feature scalling 
    sc = StandardScaler() 
    
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    unlabel = sc.fit_transform(unlabel) 
    
    return x_train, y_train, x_test, y_test, unlabel, label 

#==============================================================================
'''
DATA SELECTION STRATEGIES

rank the prediction uncertainty for each predicted sample based on the 
classification probability autput  
'''

def EntropySelection(y_probab):
    ''' Uncertainty with Entropy value'''
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
def MarginSamplingSelection(y_probab):   
    ''' Uncertainty with Margine sampling '''
    # Selecting Margine samples
    rev = np.sort(y_probab, axis=1)[:, ::-1]
    values = rev[:, 0] - rev[:, 1]
    selection = np.argsort(values)[:step]
    return selection

def RandomSelection(y_probab):
    '''Random Selection'''
    selection = np.random.choice(y_probab.shape[0], step, replace=False)
    #selection = list(selection)
    return selection

#=======================CLASSIFICATION MODELS==================================
def RF(x_train, y_train, unlabel, x_test, y_test ):
    '''Random Forest Classifier'''
    classifier = RandomForestClassifier(n_estimators=1000, n_jobs=30)
    classifier.fit(x_train, y_train)
    y_probab = classifier.predict_proba(unlabel)
    score = classifier.score(x_test, y_test)
    return (y_probab, score)
   
def LogReg(x_train, y_train, unlabel, x_test, y_test ):
    '''Logistic Regression'''
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

def GBC(x_train, y_train, unlabel, x_test, y_test):
    '''GradientBoostingClassifier'''
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

def SVM(x_train, y_train, unlabel, x_test, y_test):
    '''Support Vector Machine'''
    param = {
            'kernel': 'linear',
            'probability': True, # neccessary for AL process
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
steps = [200]
iters = 1 # number of iterations within each selection step
models = [RF]#, SVM, LogReg]#, GBC]
selection_functions = [MarginSamplingSelection]

#===================================EXPERIMENT=================================
if __name__ == '__main__':
    
    #Loading the data
    train, test = download()
    
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
            
                    # split dataset into train(10 %), test(25 %), unlabel(99.8 %) 
                    x_train, y_train, x_test, y_test, unlabel, label = split(train, test, 0.06) 
    
                    # let the active method select up to ~20% of training data
                    #max_training = round(0.026*len(label)) 
                    max_training = len(label)-200
                    
                    # create an empty list inside the dictionary for each iteration
                    acc[model.__name__+ '_' +selection_function.__name__+'_'+ str(step)][f'iter_{i}'] = []
                    
                    # record the training data for each iteration
                    acc[f'step_{step}'] = []
    
                    '''Active Selection'''
                    while len(x_train) <= max_training:
                        # probability by a classifier 
                        y_probab,_ = model(x_train, y_train, unlabel, x_test, y_test)
                        # uncertainty points by a given selection strategy
                        uncrt_pt_ind = selection_function(y_probab)
    
                        #print('x_train before Iter', x_train.shape[0])
                        x_train = np.append(unlabel[uncrt_pt_ind, :], x_train, axis = 0) 
                        y_train = np.append(label[uncrt_pt_ind], y_train) 
                        unlabel = np.delete(unlabel, uncrt_pt_ind, axis = 0) 
                        label = np.delete(label, uncrt_pt_ind) 
                            
                        # record the accuracy after every iteration
                        #score = model(x_train, y_train, x_test, y_test, unlabel)[1]
                        _, score = model(x_train, y_train, unlabel, x_test, y_test)
                        #print('x_train after Iter', x_train.shape[0])
                        #acc[model.__name__+ '_' +selection_function.__name__+'_'+ str(step)].append(score)
                        
                        acc[model.__name__+ '_' +selection_function.__name__+'_'+ str(step)][f'iter_{i}'].append(score)
                        
                        # record the training data after each iteration
                        acc[f'step_{step}'].append(len(x_train))
                        
                        #if score >= 0.75:
                        #    break
                    
            # save pickled disctionary with results
            fname = 'AL_full_experiment_large_init_agri_' + str(step) + '.pkl'
            pickle_save(fname, acc)

# recrd the experiment time
t1 = time.time()
run_time = round((t1-t0)/60, 2)

experiment_time = {}

experiment_time['run_time']= []
experiment_time['run_time'].append(run_time)

# save pickled disctionary with results
name = 'run_time_' + fname
pickle_save(name, experiment_time)


print('Training time:', round((t1-t0)/60,2), 'mins')
#==============================================================================
# Load the pickled file anc check the results
#step = 20
#acc = pickle_load('AL_full_experiment_Indian_pine_' + str(step)+'.pkl')

#acc.keys()









    
    
    