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
from sklearn.impute import SimpleImputer 
import glob

#===================
# Defining paths
#===================
#sys.path.append(join(dirname(__file__), '..', '..'))
#PATH = r'C:\Users\arman\Desktop\ActiveLearning\Experiment\dgt_T29SND'
PATH = r'C:\Users\mkhudinyan\Desktop\GitHub\active-learning\AL_T29SND'

FEATURE_NAME_PATH = join(PATH, 'feature_importance', 'feature_rankings.csv')
all_29SND_path = join(PATH, 'train_data', 'all_29SND_best_feat.csv')
LABEL_RASTER_PATH = join(PATH, 'truth_rasters', 'truth_patches_0.tif')
OUT_PATH = join(PATH, 'train_data')

SENTINEL_PATH = r'\\dgt-759\S2_2018\Theia_S2process\T29SND\composites'
METRICS_PATH = join(SENTINEL_PATH, 'metrics')
INDICES_PATH = join(SENTINEL_PATH, 'indices')
#==============================================================================
# fixing the random seed for reproducibility
seed=np.random.seed(7)
# number of samples per class to be randomly selected
# those classes are agricultural classes
n_sample = 250

#==============================
# load csv files
#==============================
# load all samples from tile 29SND
all_29SND_best_feat = pd.read_csv(all_29SND_path, index_col=0)
# load the initial train samples
train_initial_29SND = pd.read_csv(join(PATH, 'train_data', 'train_initial_29SND.csv'), index_col=0)
# subtract intial train from all samples of 29SND to 
# get the samples not included in the initial train
candidate_samples = all_29SND_best_feat.loc[~all_29SND_best_feat['Object_ID'].isin(train_initial_29SND['Object_ID'].values)]

#=============================================================================
# subsetting the training data
class_id = sorted(candidate_samples.classes.unique().tolist())
class_size = candidate_samples.groupby('classes').size()
class_size_dict = dict(class_size)

to_list = [class_size_dict[keys] for keys in class_size_dict]  

# create list containing sample size per class
sample_size = []

for i in to_list:
    if i > n_sample:
        sample_size.append(n_sample)
    else:
        sample_size.append(i)

df = pd.DataFrame({'classes':class_id, 'size':sample_size})

# sample data by given sample size
new_train_29SND =  candidate_samples.groupby('classes')\
                   .apply(lambda x: x.sample(df.loc[df['classes']== x['classes'].iloc[0], 'size'].iloc[0]))\
                   .reset_index(drop=True)

# make list of classes that are agricultural and won't be photointerpreted
agri_classes = [9,10,11,12,13,14,15,16,17,18] # 10 classes

# select only agricultural classes out of new_train_data
agri_samples = new_train_29SND.loc[new_train_29SND['classes'].isin(agri_classes)]

agri_samples.to_csv(join(PATH, 'train_data', 'agri_samples_1_sel.csv'), sep=',',header=True, index=True)

#================================================================
# Sampling the photointerpreted pixels 
#===============================================================
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
best_ind_metr_path = [path for path in metric_index if 'b3_q90-q10' not in path]


#=============================================================
# load raster files
#=============================================================
# Read the rasters as arraycand append to list
bands_list = []
bands_names = []

# load photointerpreted pixels raster
LABEL_RASTER = rasterio.open(LABEL_RASTER_PATH)
bands_list.append(LABEL_RASTER.read(1))
bands_names.append('classes')
print('Loaded raster classes')

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


# create coordinates for the pixels
x = np.arange(0, cols, 1)
y = np.arange(0, rows, 1)
xx,yy = np.meshgrid(x,y,)

bands_list.append(xx)
bands_list.append(yy)

# make a d-stack array from the bands (3D)
data_stack = np.dstack(bands_list)
_, _, n_bands = data_stack.shape
# number of pixels on one band
n_pixels = rows*(cols)
# stack bands as columns each (2D)
flat_pixels = data_stack.reshape((n_pixels, n_bands))
del bands_list
del data_stack

# Unmute in case for classification not an uncomplete tile
# drop Nones using the rasterised COS
index = np.argwhere(flat_pixels[:,0]==-1)
sampled_stack = np.delete(flat_pixels, index, axis = 0)
#xy = sampled_stack[:, [-2,-1]]
del flat_pixels

columns = bands_names + ['x','y']

df_sampled  = pd.DataFrame(sampled_stack, columns=columns)

#=========================================================
# Pass the real CRS to X and Y
#=========================================================
# set the transformation factor as Affine
transf = src.meta['transform']

# select the centeroid of the pixel.
# for this we reduce the original pixel coordinates by
# the half of the pixel size (5 m in this case)

# because the initial points are sampled from the angle
# we don't get the centroid of th epoints
#transf = Affine(10.0, 0.0, 499975.0,
#                0.0, -10.0, 4400045.0)

geo_coords = df_sampled.apply(
    lambda row: transf*(row['x'], row['y']), axis=1
)

df_sampled['x_geo'] = geo_coords.apply(lambda x: x[0])
df_sampled['y_geo'] = geo_coords.apply(lambda x: x[1])

# drop pixel coords for raster
df_sampled = df_sampled.drop(columns=['x','y'])
df_sampled.classes = df_sampled.classes.astype(int)

'''
# rearrange the columns names to match the other dataframes
'''
columns_order = agri_samples.drop(columns = 'Object_ID').columns.tolist()
df_sampled = df_sampled.loc[:, columns_order]

#=============================================================================
# subsetting the training data
class_id2 = sorted(df_sampled.classes.unique().tolist())
class_size2 = df_sampled.groupby('classes').size()
class_size_dict2 = dict(class_size2)

to_list2 = [class_size_dict2[keys] for keys in class_size_dict2]  

# create list containing sample size per class
sample_size2 = []

for i in to_list2:
    if i > n_sample:
        sample_size2.append(n_sample)
    else:
        sample_size2.append(i)

df2 = pd.DataFrame({'classes':class_id2, 'size':sample_size2})

# sample data by given sample size
active_selected_1_29SND =  df_sampled.groupby('classes')\
                   .apply(lambda x: x.sample(df2.loc[df2['classes']== x['classes'].iloc[0], 'size'].iloc[0]))\
                   .reset_index(drop=True)

active_selected_1_29SND.to_csv(join(PATH, 'train_data', 'active_selected_1_29SND.csv'), sep=',',header=True, index=True)

#================================================
# Concatinate the parts of training data into one
#================================================
train_data_1 = pd.concat((train_initial_29SND.drop(columns=['Object_ID']),\
                          agri_samples.drop(columns=['Object_ID']),\
                          active_selected_1_29SND), axis = 0,  ignore_index=True)

train_data_1.to_csv(join(PATH, 'train_data', 'train_data_1.csv'), sep=',',header=True, index=True)






















