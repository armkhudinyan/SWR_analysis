# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:17:05 2020

@author: arman
"""

import os

from statistics import mean#, stdev 
#from collections import Counter
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

def mean_acc(step):
    '''Calculate mean values for classification outputs for N iterations and
    add to the values of iterations in the dictionary'''
        
    #load the pickeld dictionary
    acc = pickle_load(pickle_name + str(step)+'.pkl')
    #acc = pickle_load(os.path.join(pickle_path, pickle_name + str(step)+'.pkl'))
    
    # calculate mean accuracy if the iterations are more than 1
    if len(acc[(list(acc))[0]]) >1:
        
        for model in models:
            for selection_function in selection_functions:
                
                # make a list of lists containing accuracies for each iteration
                to_list = [acc[f'{model}_{selection_function}_{step}'][keys] for keys in acc[f'{model}_{selection_function}_{step}']]            
                
                # calculate mean accuracies from  n iterations for each step
                mean_acc = [mean(k) for k in zip(*to_list)]
                #stdev_acc = [stdev(k) for k in zip(*to_list)]
                
                # calculate min and max values for plotting the 
                min_acc = [min(k) for k in zip(*to_list)]
                max_acc = [max(k) for k in zip(*to_list)]
                
                # append mean accuracies to the dictionary toghether with iterations' results
                acc[f'{model}_{selection_function}_{step}']['mean'] = mean_acc
                #acc[f'{model}_{selection_function}_{step}']['stdev'] = stdev_acc
                
                acc[f'{model}_{selection_function}_{step}']['min'] = min_acc
                acc[f'{model}_{selection_function}_{step}']['max'] = max_acc
    else:
        pass
    return acc


def result_plot(step, classif, selection, acc):
        
    fig, ax = plt.subplots(figsize=(8, 6))
    # plot the upper results for the classifier with standard train/test split
    ax.axhline(standard[f'standard_{classif}'], label='standard_' + str(classif))
    
    if len(acc[(list(acc))[0]]) ==1:
        # plot results for a single iteration
        ax.plot(acc[f'step_{step}'], acc[f'{classif}_{selection}_{step}']['iter_1'], label = str(selection))
        
        # plot the  accuracies for a given model and selection strategy
        #ax.plot(acc[f'step_{step}'], acc[f'{classif}_EntropySelection_{step}']['iter_1'], label = 'Entropy selection')
        #ax.plot(acc[f'step_{step}'][:87], acc[f'{classif}_MarginSamplingSelection_{step}']['iter_1'], label = 'Margin selection',
        #        C= 'r', ls = '-')
        #ax.plot(acc[f'step_{step}'], acc[f'{classif}_RandomSelection_{step}']['iter_1'], label = 'Random selection')
    else:
        # plot the mean accuracies for a given model and selection strategy
        ax.plot(acc[f'step_{step}'], acc[f'{classif}_EntropySelection_{step}']['mean'], label = 'Entropy selection')
        ax.plot(acc[f'step_{step}'], acc[f'{classif}_MarginSamplingSelection_{step}']['mean'], label = 'Margin selection')
        ax.plot(acc[f'step_{step}'], acc[f'{classif}_RandomSelection_{step}']['mean'], label = 'Random selection')
        
        # plot the min/max boundaries around the mean value
        plt.fill_between(acc[f'step_{step}'], acc[f'{classif}_EntropySelection_{step}']['min'],
                        acc[f'{classif}_EntropySelection_{step}']['max'], alpha = 0.3)
        plt.fill_between(acc[f'step_{step}'], acc[f'{classif}_MarginSamplingSelection_{step}']['min'],
                        acc[f'{classif}_MarginSamplingSelection_{step}']['max'], alpha = 0.3)
        plt.fill_between(acc[f'step_{step}'], acc[f'{classif}_RandomSelection_{step}']['min'],
                        acc[f'{classif}_RandomSelection_{step}']['max'], alpha = 0.3)
    
    ax.set_ylim([0.6,0.8])
    #ax.set_ylim([0,30000])
    ax.grid(True)
    ax.legend(loc=4, fontsize='x-large')
    
    plt.xlabel('Training data', fontsize='x-large')
    plt.ylabel('Accuracy',  fontsize='x-large')
    plt.title(f'{classif}     Step[{step}]',  fontsize='xx-large')
    plt.show()


def model_perform_plot(step, selection_function, acc):

    fig, ax = plt.subplots(figsize=(10, 8))
    # plot the upper results for the classifier with standard train/test split
    ax.axhline(standard[f'standard_RF'], label='standard_RF')
    
     # plot the mean accuracies for a diven model and selection strategy
    ax.plot(acc[f'step_{step}'], acc[str(models[0]) +'_'+ str(selection_function) +'_'+ str(step)]['iter_1'], label = models[0])
    ax.plot(acc[f'step_{step}'], acc[str(models[0]) +'_'+ str(selection_function) +'_'+ str(step)]['Iter_1'], label = models[0])
    #ax.plot(acc[f'step_{step}'], acc[str(models[2]) +'_'+ str(selection_function) +'_'+ str(step)]['mean'], label = models[2])
        
    ax.set_ylim([0.6,0.8])
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
standard = {'standard_RF':0.75, 'standard_LogReg':  0.73, 'standard_SVM': 0.74}

steps = [200] 
pickle_name = 'AL_full_experiment_large_init_agri_' # does not include steps
models = ['RF']#, 'SVM', 'LogReg']
selection_functions = [ 'MarginSamplingSelection']#, 'RandomSelection']#,'EntropySelection']

# calculate the mean, min and max value of N iterations
#acc_10 = mean_acc(10)
#acc_20 = mean_acc(20)
#acc_40 = mean_acc(40)
acc_200 = mean_acc(200)

# plot the reuslts for Random Forest
#result_plot(10, classif= 'RF', acc = acc_10 )    
#result_plot(20, classif= 'RF', acc = acc_20 )
#result_plot(40, classif= 'RF', acc = acc_40 )
#result_plot(200, classif= 'RF',selection = 'MarginSamplingSelection', acc = acc_200 )

for selection_function in selection_functions:
    for step in steps:
        result_plot(step, classif= 'RF',selection = selection_function , acc = acc_200 )

'''
# plot the results for Logistic Regression 
result_plot(10, classif= 'LogReg', acc=acc_10 )    
result_plot(20, classif= 'LogReg', acc=acc_20 )
result_plot(40, classif= 'LogReg', acc=acc_40 )

# plot the results for SVM
result_plot(10, classif= 'SVM', acc=acc_10 )    
result_plot(20, classif= 'SVM', acc=acc_20 )
result_plot(40, classif= 'SVM', acc=acc_40 )

# plot the full experimental run results
model_perform_plot(20, selection_function = selection_functions[0], acc=acc_20)
model_perform_plot(20, selection_function = selection_functions[1], acc=acc_20)
model_perform_plot(20, selection_function = selection_functions[2], acc=acc_20)
'''

#==============================================================================
# plot the initialization experiment results
#==============================================================================

acc_small = pickle_load('AL_small_init_agri_200.pkl')
acc_large = pickle_load('AL_large_init_agri_200.pkl')

fig, ax = plt.subplots(figsize=(8, 6))
# plot the upper results for the classifier with standard train/test split
ax.axhline(standard[f'standard_RF'], label='standard_RF')
#ax.axhline(0.683, label='standard_RF')

ax.plot(acc_small['step_200'], acc_small[f'RF_MarginSamplingSelection_200']['iter_1'], label = 'Margin sel small init')
ax.plot(acc_large['step_200'][:87], acc_large[f'RF_MarginSamplingSelection_200']['iter_1'], label = 'Margin sel large init',
        C= 'r', ls = '-')
#ax.plot(acc_large['step_200'], acc_large[f'RF_RandomSelection_200']['iter_1'], label = 'Random selection')

ax.set_ylim([0.4,0.9])
ax.grid(True)
ax.legend(loc=4, fontsize='x-large')

plt.xlabel('Training data', fontsize='x-large')
plt.ylabel('Accuracy',  fontsize='x-large')
plt.title(f'RF    Step[200]',  fontsize='xx-large')
plt.show()




acc_large['RF_MarginSamplingSelection_200']['iter_1'][0]
acc_small['RF_MarginSamplingSelection_200']['iter_1'][23]

x_large_init = acc_large['step_200'][0]-200  # 11588
x_large_max = acc_large['step_200'][86]  # 28988

x_small_init = acc_small['step_200'][0]-200  # 231
x_sall_max = acc_small['step_200'][-1]   # 24631

small_eq_large = acc_small['RF_MarginSamplingSelection_200']['iter_1'][23]
train_small_eq = acc_small['step_200'][23] # 5031


Al_small = x_sall_max - x_small_init # 6557

Al_large = x_large_max - x_large_init 




