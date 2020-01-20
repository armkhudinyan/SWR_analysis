# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:17:05 2020

@author: arman
"""

import os

from statistics import mean 
from collections import Counter
import pickle
import matplotlib.pyplot as plt

from statistics import stdev

#==============================================================================
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
'''Classification accuracies of standard train-test split
(test_size = 0.33). The results are the average of 5 iterations'''

def mean_acc(step):
    '''Calculate mean values for classification outputs for N iterations and
    add to the values of iterations in the dictionary'''
        
    #load the pickeld dictionary
    acc = pickle_load(pickle_name + str(step)+'.pkl')
    
    for model in models:
        for selection_function in selection_functions:
            
            
            # make a list of lists containing accuracies for each iteration
            to_list = [acc[f'{model}_{selection_function}_{step}'][keys] for keys in acc[f'{model}_{selection_function}_{step}']]            
            
            # calculate mean accuracies from  n iterations for each step
            mean_acc = [mean(k) for k in zip(*to_list)]
            
            # append mean accuracies to the dictionary toghether with iterations' results
            acc[f'{model}_{selection_function}_{step}']['mean'] = mean_acc
            
    return acc


def result_plot(step, classif, acc):
    #acc= pickle_load('AL_full_experiment_Indian_pine_' + str(step)+'.pkl')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    #ax.axhline(standard['standard_'+ str(classif)], label='standard_' + str(classif))
    ax.axhline(standard[f'standard_{classif}'], label='standard_' + str(classif))
       
    ax.plot(acc[f'step_{step}'], acc[f'{classif}_EntropySelection_{step}']['mean'], label = 'Entropy selection')
    ax.plot(acc[f'step_{step}'], acc[f'{classif}_MarginSamplingSelection_{step}']['mean'], label = 'Margin selection')
    ax.plot(acc[f'step_{step}'], acc[f'{classif}_RandomSelection_{step}']['mean'], label = 'Random selection')
        
    ax.set_ylim([0.6,1])
    ax.grid(True)
    ax.legend(loc=4, fontsize='x-large')
    
    plt.xlabel('Training data', fontsize='x-large')
    plt.ylabel('Accuracy',  fontsize='x-large')
    plt.title(f'{classif}     Step[{step}]',  fontsize='xx-large')
    plt.show()


def model_perform_plot(step, selection_function, acc):

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axhline(standard[f'standard_RF'], label='standard_RF')
       
    ax.plot(acc[f'step_{step}'], acc[str(models[0]) +'_'+ str(selection_function) +'_'+ str(step)]['mean'], label = models[0])
    ax.plot(acc[f'step_{step}'], acc[str(models[1]) +'_'+ str(selection_function) +'_'+ str(step)]['mean'], label = models[1])
    ax.plot(acc[f'step_{step}'], acc[str(models[2]) +'_'+ str(selection_function) +'_'+ str(step)]['mean'], label = models[2])
        
    ax.set_ylim([0.65,1])
    ax.grid(True)
    ax.legend(loc=4, fontsize='x-large')
    
    plt.xlabel('Training data size', fontsize='x-large')
    plt.ylabel('Accuracy',  fontsize='x-large')
    plt.title(f'{selection_function}      Step[{step}]',  fontsize='xx-large')
    plt.show()


#==============================================================================
# Plot calculate the 'mean' accuracy and plot the graphs
#==============================================================================
'''
Standard classifiers with train/test split (test_size= 0.33) were shuffeled
5 times and the results are averaged
'''
#standard_RF= 0.91
#standard_LogReg= 0.90
#standard_SVM= 0.90
standard = {'standard_RF':0.91, 'standard_LogReg':  0.90, 'standard_SVM': 0.90}

#step = 20 
pickle_name = 'AL_full_experiment_Indian_pine_' # does not include steps
models = ['RF', 'SVM', 'LogReg']
selection_functions = [ 'MarginSamplingSelection','EntropySelection', 'RandomSelection']

# calculate the mean value of N iterations
acc_10 = mean_acc(10)
acc_20 = mean_acc(20)
acc_40 = mean_acc(40)


# plot the reuslts for Random Forest
result_plot(10, classif= 'RF', acc = acc_10 )    
result_plot(20, classif= 'RF', acc = acc_20 )
result_plot(40, classif= 'RF', acc = acc_40 )

# plot the reuslts for Logistic Regression 
result_plot(10, classif= 'LogReg', acc=acc_10 )    
result_plot(20, classif= 'LogReg', acc=acc_20 )
result_plot(40, classif= 'LogReg', acc=acc_40 )

# plot the reuslts for SVM
result_plot(10, classif= 'SVM', acc=acc_10 )    
result_plot(20, classif= 'SVM', acc=acc_20 )
result_plot(40, classif= 'SVM', acc=acc_40 )

# plot the full experimental run results
model_perform_plot(20, selection_function = selection_functions[0], acc=acc_20)
model_perform_plot(20, selection_function = selection_functions[1], acc=acc_20)
model_perform_plot(20, selection_function = selection_functions[2], acc=acc_20)









