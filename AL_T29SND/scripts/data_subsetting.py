# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:06:33 2020

@author: arman
"""

import time
import os
from os.path import join, dirname

import numpy as np
import pandas as pd

import rasterio
from rasterio.plot import show
from rasterio import features
from rasterio import Affine 
import shapely
import geopandas as gpd
import gdal
import ogr
from shapely.geometry import Point
from sklearn.impute import SimpleImputer 

import glob
from collections import Counter
import pickle
#===================
# Defining paths
#===================
#sys.path.append(join(dirname(__file__), '..', '..'))
PATH = r'C:\Users\arman\Desktop\ActiveLearning\Experiment\dgt_T29SND'

all_train_path = join(PATH, 'train_data', 'sampled_point_coordinates_s2.csv')
xy_29SND_path = join(PATH, 'train_data', 'sample_xy_29SND.csv')
out_path = join(PATH, 'train_data')

#==============================================================================
# Define functions for saving and loading pickles

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
# fixt the start time
t0 = time.time()
# fixing the random seed for reproducibility
seed=np.random.seed(7)
# number of samples per class to be randomly selected
n_sample = 250

#==============================
# load csv files
#==============================
# make a list with all raster's paths
all_train = pd.read_csv(all_train_path, index_col=0)
class_names = all_train.class_decoded.unique().tolist()
print('Classes', class_names)
xy_29SND = pd.read_csv(xy_29SND_path)

xy_29SND = xy_29SND.drop(columns =['class', 'y_geo', 'x_geo', 'class_deco'])
#xy_29SND = xy_29SND.loc[:,'ObjectedID'].copy()

sampled_29SND = xy_29SND.set_index('ObjectedID')\
                .join(all_train.set_index('ObjectedID'), lsuffix='_l', rsuffix='_r', how = 'left')\
                .rename(columns={'class':'classes'})\
                .reset_index()
                  
# save to csv
sampled_29SND.to_csv(join(out_path,"sampled_29SND.csv"), sep=',',header=True, index=True)

# subsetting the training data
class_id = sampled_29SND.classes.unique()
class_size = sampled_29SND.groupby('classes').size()
class_size_dict = dict(class_size)

to_list = [class_size_dict[keys] for keys in class_size_dict]  

# create list containing sample size per class
sample_size = []

for i in to_list:
    if i > n_sample:
        sample_size.append(n_sample)
    else:
        sample_size.append(i)

df_sample_size = pd.Series(sample_size)

# sample data by given sample size
train_29SND = sampled_29SND.groupby('classes')\
               .apply(lambda x: x.sample(df_sample_size[x['classes'].iloc[0]]))\
               .reset_index(drop=True)

# save training samples
train_29SND.to_csv(join(out_path,"train_29SND.csv"), sep=',',header=True, index=True)


#==============================
# Select The most important features for classification
#==============================
feat_ranking = pd.read_csv(join(PATH, 'feature_importance', 'feature_rankings.csv'))
train_29SND = pd.read_csv(join(out_path, 'train_29SND.csv'))

# list of best 40 featurs names
best_feat = feat_ranking['feature'].tolist()[:40]
train_29SND_best_feat = train_29SND.loc[:, ['classes']+best_feat]
# save training data with best features
train_29SND_best_feat.to_csv(join(out_path, 'train_best_40.csv'))


# recrd the experiment time
t1 = time.time()
run_time = round((t1-t0)/60, 2)

experiment_time = {}

experiment_time['run_time']= []
experiment_time['run_time'].append(run_time)

# save pickled disctionary with results
name = 'run_time_sampling'
#pickle_save(name, experiment_time)


print('Training time:', round((t1-t0)/60,2), 'mins')
