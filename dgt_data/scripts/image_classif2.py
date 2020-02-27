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
import gdal
import ogr
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer 

import glob
from collections import Counter
import pickle
#===================
# Defining paths
#===================
#sys.path.append(join(dirname(__file__), '..', '..'))
PATH = r'C:\Users\mkhudinyan\Desktop\Arman\dgt'
SENTINEL_PATH = r'G:\DGT_data'


s1_path = join(SENTINEL_PATH, 'sentinel1_clipped')
s2_path = join(SENTINEL_PATH, 'sentinel2_clipped')
buffered_cos_path = join(PATH, 'data')
shape_path = join(PATH,  'cos_shape', 'COS_2015_clipped.shp')
csv_path = join(PATH, 'data','train_sampled.csv')

out_path = join(PATH, 'data')

# raster band's names
BANDS_NAME = {
            'BANDS_NAME_S2_N1' :  ['B2',  'B3',  'B4',   'B5',   'B8'  ],
            'BANDS_NAME_S2_N2' :  ['B8A', 'B11', 'NDVI', 'NDMI', 'NDBI'],
            'BANDS_NAME_S1'    :  ['fallVV', 'winterVV', 'springVV', 'summerVV', 
                                  'fallVH', 'winterVH', 'springVH', 'summerVH'],
            'BANDS_NAME_S1_GLCM': ['savg', 'ent', 'corr', 'idm', 'var', 'diss', 'contrast', 'asm']
            }

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
def RF(x_train, y_train, image_array):
    '''Random Forest Classifier'''
    classifier = RandomForestClassifier(n_estimators=500)
    classifier.fit(x_train, y_train)
    y_probab = classifier.predict_proba(image_array)
    predicted = classifier.predict(image_array)
    return (y_probab, predicted)

#=======================UNCERTAINTY MAPPING====================================
def MarginSamplingSelection(y_probab):
    ''' Uncertainty with Margine sampling '''
    # Selecting Margine samples as a smallest difference of probability values
    # between the first and second most probabel classes
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
# load raster files
#==============================
# make a list with all raster's paths
rasters = []

search_criteria = "*.tif"
for path in (s2_path, s1_path):
    querie = join(path, search_criteria)   
    raster_path = glob.glob(querie)
    rasters.extend(raster_path)
# choose subset of the rasters
rasters =  [rasters[index] for index in [2]]

# Read the label and rs imagery rasters as array
bands_list = []

labels = rasterio.open(join(buffered_cos_path, 'buffer_20_cos2015.tif'))
bands_list.append(labels.read(1))

# read rasters as array, fill missing values and appent to the list
imputer = SimpleImputer(missing_values = np.nan, strategy ="mean")

for raster in rasters:
    src = rasterio.open(raster)
    out_meta = src.meta.copy()
    proj = src.crs
    transform = src.get_transform()
    count = src.count 
    for b in range(1, count + 1):
        # read a single band as 2D array
        band = src.read(b)
        rows, cols = band.shape
        # imputing missing data 
        imputer = imputer.fit(band) 
        band = imputer.transform(band) 

        bands_list.append(band)
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
n_pixels = rows*cols
# stack bands as columns each (2D)
flat_pixels = data_stack.reshape((n_pixels, n_bands))
del data_stack

# drop Nones using the rasterised COS
index = np.argwhere(flat_pixels[:,0]==-1)
all_stack = np.delete(flat_pixels, index, axis = 0)
xy = all_stack[:, [-2,-1]]

del flat_pixels

#==============================
# Loading the training data
#==============================
df = pd.read_csv(csv_path)
df = df.drop('Unnamed: 0')

# filling the missing values
#imputer = SimpleImputer(missing_values = np.nan, strategy ="mean") 

x_train = df.iloc[:, 1:11].values
imputer1 = imputer.fit(x_train) 
x_train = imputer1.transform(x_train) 

y_train = df.iloc[:, 0].values

image_array = all_stack[:, 1:11] # unmute in case of xy coordinets
#imputer2 = imputer.fit(image_array) 
#image_array = imputer2.transform(image_array) 


#==============================
# Train RF model and classify image
#==============================
y_probab, predicted = RF(x_train, y_train, image_array)

# Mapping the uncertainty
uncertainty = MarginSamplingSelection(y_probab)

#==============================
# Save classification and uncertainty maps
#==============================
# make dataframe from arrays containing raster x and y 
columns1=['pred','x','y']
columns2=['uncert','x','y']

# define empty DataFrame with true raster size
df_origin_pred = pd.DataFrame(flat_pixels, columns=columns1)
df_origin_pred.pred = np.nan
df_origin_pred = df_origin_pred.set_index(['x','y'])

df_origin_uncert = pd.DataFrame(flat_pixels, columns=columns2)
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

#=====================
# Write tif files
#=====================
#rasterize using the shape and transform of the satellite image
out_meta  = src.meta

out_meta.update({
               "driver": "GTiff",
               "dtype" : 'float32',
               "nodata": None, #np.nan,
               "height": src.height,
               "width" : src.width,
               "transform": transform,
               "count" : 1,
               #to any crs by defining here. if you don't do so, 
        		   # it sets up to wgs84 geographic by default (weird tho)
               #"crs": "+proj=utm +zone=29 +ellps=WGS84 +datum=WGS84 +units=m +no_defs "
               "crs":  out_meta['crs']
              })
          
# Write the raster to disk
with rasterio.open(join(out_path, 'data', 'lulc_1.tif'), "w", **out_meta) as dest:
    dest.write(classif_map, indexes=1)  

with rasterio.open(join(out_path, 'data', 'uncertainty_1.tif'), "w", **out_meta) as dest:
    dest.write(uncert_map, indexes=1)  


'''
# Output the simulated landcover raster
def write_geotiff(dest_path, data, geo_transform, projection):
    #"""Create a GeoTIFF file with the given data."""
    driver= gdal.GetDriverByName('GTiff')
    rows, cols= data.shape
    dataset= driver.Create(dest_path, cols, rows, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band= dataset.GetRasterBand(1)
    band.WriteArray(data)
    dataset=None
    
# write and close the file for 2028 Business as usual
write_geotiff(dest_path = join(out_path, 'data', 'lulc_1.tif'),        classif_map, transform, proj)
write_geotiff(dest_path = join(out_path, 'data', 'uncertainty_1.tif'), uncert_map,  transform, proj)
'''

# recrd the experiment time
t1 = time.time()
run_time = round((t1-t0)/60, 2)

experiment_time = {}

experiment_time['run_time']= []
experiment_time['run_time'].append(run_time)

# save pickled disctionary with results
name = 'run_time_sampling'
pickle_save(name, experiment_time)


print('Training time:', round((t1-t0)/60,2), 'mins')
































