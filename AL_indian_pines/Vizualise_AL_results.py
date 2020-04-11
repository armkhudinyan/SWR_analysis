# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:17:05 2020

@author: arman
"""

import os

from statistics import mean, stdev 
from collections import Counter
import pickle
import matplotlib.pyplot as plt

#PATH = r'C:\Users\arman\Desktop\ActiveLearning'
#pickle_path = os.path.join(PATH, 'Experiment', 'Arcived_pickles')
#==============================================================================
# write and open pickled files
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
# define plotting functions
#==============================================================================
def calculate_stat(step, experiment_name):
    '''Calculate mean values for classification outputs for N iterations and
    add to the values of iterations in the dictionary'''
        
    # load accuracy results from the pickeld dictionary
    acc = pickle_load(experiment_name + str(step)+'.pkl')
       
    # Calculate mean, min and max accuracies if the iterations are more than 1
    if len(acc[(list(acc))[0]]) >1:
        
        for model in models_str:
            for selection_function in selection_functions_str:
  
                # Make a list of lists containing accuracies for each iteration
                to_list = [acc[f'{model}_{selection_function}_{step}'][keys] for keys in acc[f'{model}_{selection_function}_{step}']]            
                
                # calculate mean accuracies from  n iterations for each step
                mean_acc = [mean(k) for k in zip(*to_list)]
                #stdev_acc = [stdev(k) for k in zip(*to_list)]
                
                # calculate min and max values for plotting 
                min_acc = [min(k) for k in zip(*to_list)]
                max_acc = [max(k) for k in zip(*to_list)]
                
                # append calculated accuracies to the dictionary 
                acc[f'{model}_{selection_function}_{step}']['mean'] = mean_acc
                #acc[f'{model}_{selection_function}_{step}']['stdev'] = stdev_acc
                
                acc[f'{model}_{selection_function}_{step}']['min'] = min_acc
                acc[f'{model}_{selection_function}_{step}']['max'] = max_acc
    else:
        pass
    return acc

def result_plot(step, model, selection_functions, acc):
    #sns.set_style('white')
    fig, ax = plt.subplots(figsize=(8, 6))

    if len(acc[(list(acc))[0]]) ==1: 
        
        for selection in selection_functions_str:
            # Plot results for a single iteration
            ax.plot(acc[f'step_{step}'], acc[f'{model}_{selection}_{step}']['iter_1'], label = str(selection))
    else:
        for selection in selection_functions_str:
            
            # Plot the mean accuracies for a given model and selection strategy
            ax.plot(acc[f'step_{step}'], acc[f'{model}_{selection}_{step}']['mean'], label = str(selection))

            # Plot the min/max boundaries around the mean value
            plt.fill_between(acc[f'step_{step}'], acc[f'{model}_{selection}_{step}']['min'],
                            acc[f'{model}_{selection}_{step}']['max'], alpha = 0.3)

    # plot the upper results for the classifier with standard train/test split
    ax.axhline(standard[f'standard_{model}'], label=f'standard_{model}')

    ax.set_ylim([0.5,1])
    ax.grid(True)
    ax.legend(loc=4, fontsize='x-large')
    
    plt.xlabel('Training data', fontsize='x-large')
    plt.ylabel('Accuracy',  fontsize='x-large')
    plt.title(f'{model}     Step[{step}]',  fontsize='xx-large')
    plt.show()

def model_perform_plot(step, selection_function, acc):
    '''Plot the models with same selection
    function together'''
    fig, ax = plt.subplots(figsize=(10, 8))
    # plot the upper results for the classifier with standard train/test split
    ax.axhline(standard[f'standard_RF'], label='standard_RF')
    
     # plot the mean accuracies for a diven model and selection strategy
    ax.plot(acc[f'step_{step}'], acc[str(models_str[0]) +'_'+ str(selection_function) +'_'+ str(step)]['mean'], label = models_str[0])
    ax.plot(acc[f'step_{step}'], acc[str(models_str[1]) +'_'+ str(selection_function) +'_'+ str(step)]['mean'], label = models_str[1])
    ax.plot(acc[f'step_{step}'], acc[str(models_str[2]) +'_'+ str(selection_function) +'_'+ str(step)]['mean'], label = models_str[2])
        
    ax.set_ylim([0.4,1])
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
standard = {'standard_RF':0.91, 'standard_LogReg': 0.90, 'standard_SVM': 0.90}

# Set up AL combinations for plotting
experiment_name = 'AL_full_experiment_Indian_pine_'
steps_str = ['10', '20', '40']
models_str = ['RF', 'LogReg', 'SVM']
selection_functions_str = ['MarginSamplingSelection', 'EntropySelection', 'RandomSelection']

# Calculate the statistics for N iterations
acc_10 = calculate_stat(10, experiment_name)
acc_20 = calculate_stat(20, experiment_name)
acc_40 = calculate_stat(40, experiment_name)

accuracies = [acc_10, acc_20, acc_40]


# Plot the lines of average accuracies with min-max values
for step, acc in zip(steps_str, accuracies):
    for model in models_str:
        result_plot(step, model, selection_functions_str, acc)         

# plot the full experimental run results
model_perform_plot(20, selection_function = selection_functions_str[0], acc=acc_20)
model_perform_plot(20, selection_function = selection_functions_str[1], acc=acc_20)
model_perform_plot(20, selection_function = selection_functions_str[2], acc=acc_20)
