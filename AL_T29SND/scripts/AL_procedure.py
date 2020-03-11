# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:06:08 2020

@author: arman
"""

import time
import os
import sys
from os.path import join, dirname

import numpy as np
import pandas as pd

import rasterio
from rasterio.plot import show
from rasterio import features
from rasterio import Affine 
#import geopandas as gpd
#import gdal
#import ogr
#from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer 

import glob
from collections import Counter
import pickle

# define the sequential number of AL procedure
n_run = 1
#===================
# Defining paths
#===================
#sys.path.append(join(dirname(__file__), '..', '..'))
PATH = r'C:\Users\mkhudinyan\Desktop\GitHub\active-learning\AL_T29SND'

SENTINEL_PATH = r'\\dgt-759\S2_2018\Theia_S2process\T29SND\composites'
METRICS_PATH = join(SENTINEL_PATH, 'metrics')
INDICES_PATH = join(SENTINEL_PATH, 'indices')
#INDICES_METRICS_PATH = join(SENTINEL_PATH, 'indices', 'metrics')
TRAIN_PATH = join(PATH, 'train_data','train_data_1.csv')
FEATURE_NAME_PATH = join(PATH, 'feature_importance', 'feature_rankings.csv')
OUT_PATH = join(PATH, 'results')

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

#=======================CLASSIFICATION MODELS==================================
def RF(X_train, y_train, image_array):
    '''Random Forest Classifier'''
    classifier = RandomForestClassifier(n_estimators=300, n_jobs=34)
    classifier.fit(X_train, y_train)
    y_probab = classifier.predict_proba(image_array)
    predicted = classifier.predict(image_array)
    return (y_probab, predicted)

#=======================UNCERTAINTY MAPPING====================================
def MarginSamplingSelection(y_probab):
    ''' Uncertainty with Margine sampling 
    Selecting Margine samples as a smallest difference of probability values
    between the first and second most probabel classes
    '''
    rev = np.sort(y_probab, axis=1)[:, ::-1]
    uncertainty = rev[:, 0] - rev[:, 1]
    #selection = np.argsort(values)[:step]
    return uncertainty

#==============================================================================
# fixt the start time
t0 = time.time()
# fixing the random seed for reproducibility
seed=np.random.seed(7)

#==============================
# Define the proper paths for rasters
#==============================
# make a list containg the best ranked features
feat_ranking = pd.read_csv(FEATURE_NAME_PATH)
best_feat = feat_ranking['feature'].tolist()[:40]
# change name from Portugal to the cpecified tile
best_feat = [path.replace('Portugal','T29SND') for path in best_feat]

# categorize the rasters
best_bands   = [band_name for band_name in best_feat if "." in band_name]
best_bands.sort()
best_metrics = [metric_name for metric_name in best_feat if 'q' in metric_name]
best_indices = [index_name for index_name in best_feat if '_N' in index_name]

# make a list with combined index and metrics raster's paths
rasters_path = []

search_criteria = "*.tif"
for path in ( INDICES_PATH, METRICS_PATH):
    querie = join(path, search_criteria)   
    raster_path = glob.glob(querie)
    #print(raster_path)
    rasters_path.extend(raster_path)

# choose subset of the rasters
matchers = best_metrics + best_indices
metric_index = [path for path in rasters_path if any(i in path for i in matchers)]
best_ind_metr_path = [path for path in metric_index if 'b3_q90-' not in path] # drop extra selected band

#==============================================================================
'''
# define bands path
bands_path = []

search_criteria = "*.tif"
for path in (SENTINEL_PATH,):
    querie = join(path, search_criteria)   
    raster_path = glob.glob(querie)
    bands_path.extend(raster_path)
''' 

#==============================
# load raster files
#==============================
# Read the rasters as arraycand append to list
bands_list = []
bands_names = []

# load indices and metrics arrays
for raster in best_ind_metr_path:
    src = rasterio.open(raster)
    out_meta = src.meta.copy()
    proj = src.crs
    transform = src.get_transform()
    rows, cols = src.shape
    band_name = raster.split('\\')[-1].split('.')[0]
    band = src.read(1)
    bands_list.append(band)
    bands_names.append(band_name)
    print('Loaded raster', raster.split('\\')[-1])
    del band

# load bends arrays
for band_name in best_bands:
    search = f'{band_name.split(".")[0]}.tif'
    band_n = int(band_name.split(".")[1])
    ras_path = join(SENTINEL_PATH, search)
    #ras_path = glob.glob(query)
    src = rasterio.open(ras_path)
    band = src.read(band_n)
    bands_list.append(band)
    bands_names.append(band_name)
    print('Loaded raster', band_name.split('\\')[-1])
    del band

'''
# create coordinates for the pixels
x = np.arange(0, cols, 1)
y = np.arange(0, rows, 1)
xx,yy = np.meshgrid(x,y,)

bands_list.append(xx)
bands_list.append(yy)
'''
# make a d-stack array from the bands (3D)
data_stack = np.dstack(bands_list)
_, _, n_bands = data_stack.shape
# number of pixels on one band
n_pixels = rows*(cols)
# stack bands as columns each (2D)
flat_pixels = data_stack.reshape((n_pixels, n_bands))
del bands_list
del data_stack

# fill missing values 
#df_stack = pd.DataFrame(flat_pixels, columns=bands_names)

#df_stack = pd.DataFrame(flat_pixels).fillna(method='bfill', axis=0)
#flat_pixels = df_stack.values
#del data_stack

# Unmute in case for classificationnot an uncomplete tile
# drop Nones using the rasterised COS
#index = np.argwhere(flat_pixels[:,0]==-1)
#all_stack = np.delete(flat_pixels, index, axis = 0)
#xy = all_stack[:, [-2,-1]]
#del flat_pixels

#==============================
# Loading the training data
#==============================
train_data = pd.read_csv(TRAIN_PATH, index_col=0)
'''
# changing names from Portugal T29SND
colnames = list(train_data.columns )
new_colnames = [name.replace('Portugal','T29SND') for name in colnames]
train_data = pd.DataFrame(train_data.values, columns = new_colnames)
'''
# reorder the colums names to ,atch with loaded bands order
train_data = train_data.drop(columns = ['y_geo', 'x_geo'])
train_data = train_data.loc[:, ['classes'] + bands_names]

X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values

#==============================
# Split data into subsets 
# Train RF model and classify image
#==============================
#In order to avoind from memory error, the long array can be splittet
#into subsets, then classified separatelly and reshapped into one in the end

flat_pixels_subsets = np.array_split(flat_pixels, 40)
# The second argument is the number of subsets. If it still doesn't fit
# into memory you may need to increase this number.
# The output is a list of arrays

pred_results = []
uncert_results = []

i = 1
for subset in flat_pixels_subsets:
    y_probab, predicted = RF(X_train, y_train, subset)
    uncert = MarginSamplingSelection(y_probab)
    pred_results.append(predicted)
    uncert_results.append(uncert)
    print(f'subset {i} is classified')
    i +=1

classification = np.concatenate(pred_results)
uncertainty = np.concatenate(uncert_results)

'''
#==============================
# Recover the raster shape in case 
# classifying an uncomplete tile
#==============================

# make dataframe from arrays containing raster x and y 
columns1=['pred','x','y']
columns2=['uncert','x','y']

# define empty DataFrame with true raster size
df_origin_pred = pd.DataFrame(predicted, columns=columns1)
df_origin_pred.pred = np.nan
df_origin_pred = df_origin_pred.set_index(['x','y'])

df_origin_uncert = pd.DataFrame(uncertainty, columns=columns2)
df_origin_uncert.pred = np.nan
df_origin_uncert = df_origin_uncert.set_index(['x','y'])
#df_origin2 = df_origin.rename(columns = {'pred':'uncert'}, inplace = True)


# recover raster for prediction
#predicted = all_stack[:,0]
pred_stack = np.hstack([np.expand_dims(predicted, -1), xy])
df_pred = pd.DataFrame(pred_stack,columns=columns1) # shorter list
df_pred = df_pred.set_index(['x','y'])
df_pred = df_origin_pred.join(df_pred, lsuffix = '_old', rsuffix = '_new', how='outer')

# recover raster for uncertainty
uncert_stack =  np.hstack([np.expand_dims(uncertainty, -1), xy])
df_uncert = pd.DataFrame(uncert_stack, columns=columns2) # shorter list
df_uncert = df_uncert.set_index(['x','y'])
df_uncert = df_origin_uncert.join(df_uncert, lsuffix = '_old', rsuffix = '_new', how='outer')

# reshape the vector raster to its original state to form the raster
classif_map = np.reshape(df_pred.loc['pred_new'].values, (rows, cols))
uncert_map  = np.reshape(df_uncert.loc['uncert_new'].values, (rows, cols))
'''

# in case there is no missing values and output array is equal to raster array
classif_map = np.reshape(classification.astype('float32'), (rows, cols))
uncert_map  = np.reshape(uncertainty.astype('float32'), (rows, cols))

#=====================
# Write tif files
#=====================
#rasterize using the shape and transform of the satellite image
out_meta  = src.meta.copy()

out_meta.update({
               "driver": "GTiff",
               "dtype" : 'float32',
               "nodata": None, #np.nan,
               "height": src.height,
               "width" : src.width,
               "transform": src.transform,
               "count" : 1,
               #to any crs by defining here. if you don't do so, 
        		   # it sets up to wgs84 geographic by default (weird tho)
               #"crs": "+proj=utm +zone=29 +ellps=WGS84 +datum=WGS84 +units=m +no_defs "
               "crs":  out_meta['crs']
              })
          
# Write the array to raster GeoTIF 
with rasterio.open(join(OUT_PATH, f'lulc_{n_run}.tif'), "w", **out_meta) as dest:
    dest.write(classif_map, indexes=1)  

with rasterio.open(join(OUT_PATH, f'uncertainty_{n_run}.tif'), "w", **out_meta) as dest:
    dest.write(uncert_map, indexes=1)  

#=====================
# Save training time
#=====================
# recrd the experiment time
t1 = time.time()
run_time = round((t1-t0)/60, 2)

experiment_time = {}

experiment_time['run_time']= []
experiment_time['run_time'].append(run_time)

# save pickled disctionary with results
name = f'run_time_{n_run}'
pickle_save(name, experiment_time)

print('Training time:', round((t1-t0)/60,2), 'mins')
































