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
import shapely
import geopandas as gpd
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
PATH = r'C:\Users\arman\Desktop\ActiveLearning\Experiment\my_data'
SENTINEL_PATH = r'G:\DGT_data'


s1_path = join(PATH, 'sentinel1')
s2_path = join(PATH, 'sentinel2')
buffered_cos_path = join(PATH, 'data')
shape_path = join(PATH,  'cos_shape', 'COS_2015_clipped.shp')
csv_path = join(PATH, 'data','train_sampled.csv')

out_path = join(PATH, 'data')

# raster band's names
BANDS_NAME = {
            'S2_N1' :  ['B2',  'B3',  'B4',   'B5',   'B8'  ],
            'S2_N2' :  ['B8A', 'B11', 'NDVI', 'NDMI', 'NDBI'],
            'S1'    :  ['fallVV', 'winterVV', 'springVV', 'summerVV', 
                                  'fallVH', 'winterVH', 'springVH', 'summerVH'],
            'S1_GLCM': ['savg', 'ent', 'corr', 'idm', 'var', 'diss', 'contrast', 'asm']
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
    classifier = RandomForestClassifier(n_estimators=500, n_jobs=-1)
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
rasters =  [rasters[index] for index in [1]]

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
    #for b in range(1, count + 1):
    for b in [1,2]:
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
del flat_pixels

#==============================
# Loading the training data
#==============================
df = pd.read_csv(csv_path)
df = df.drop('Unnamed: 0', axis=1)

# filling the missing values
imputer = SimpleImputer(missing_values = np.nan, strategy ="mean") 

x_train = df.iloc[:, 6:8].values
imputer1 = imputer.fit(x_train) 
x_train = imputer1.transform(x_train) 

y_train = df.iloc[:, 0].values

image_array = all_stack[:, 1:3] # unmute in case of xy coordinets
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
columns = ['classes'] + BANDS_NAME['S2_N2'][0:2] + ['x','y']
     
# Make a DataFrame from the long_stacked images
df_coords = pd.DataFrame(all_stack, columns=columns)

#==============================
# Pass the real CRS to X and Y
#==============================
# set the transformation factor as Affine
transf = src.meta['transform']

# select the centeroid of the pixel.
# for this we reduce the original pixel coordinates by
# the half of the pixel size (5 m in this case)
transf = Affine(10.0, 0.0, 499980.0,
               0.0, -10.0, 4400050.0)

geo_coords = df_coords.apply(
    lambda row: transf*(row['x'], row['y']), axis=1
)


df_coords['x_geo'] = geo_coords.apply(lambda x: x[0])
df_coords['y_geo'] = geo_coords.apply(lambda x: x[1])

# drop pixel coords for raster
df_coords = df_coords.drop(columns=['x','y'])

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
write_geotiff(dest_path = join(out_path, 'data', 'lulc1'),        classif_map, transform, proj)
write_geotiff(dest_path = join(out_path, 'data', 'uncertainty1'), uncert_map,  transform, proj)



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






bands_list2 = []

labels = rasterio.open(join(buffered_cos_path, 'buffer_1_cos2015.tif'))
bands_list2.append(labels.read(1))

rows, cols =  labels.read(1).shape

x = np.arange(0, cols, 1)
y = np.arange(0, rows, 1)
xx,yy = np.meshgrid(x,y,)

bands_list2.append(xx)
bands_list2.append(yy)

data_stack2 = np.dstack(bands_list2)

_, _, n_bands = data_stack2.shape
# number of pixels on one band
n_pixels = rows*cols
# stack bands as columns each (2D)
flat_pixels2 = data_stack2.reshape((n_pixels, n_bands))
del data_stack2

# drop Nones using the rasterised COS
index = np.argwhere(flat_pixels2[:,0]==-1)
all_stack = np.delete(flat_pixels2, index, axis = 0)
#del flat_pixels

columns=['classes','x','y']


df_0 = pd.DataFrame(all_stack,columns=columns) # shorter list
df_0 = df_0.set_index(['x','y'])

df_2 = pd.DataFrame(flat_pixels2, columns=columns) # full raster's list
df_2.classes = np.nan
df_2 = df_2.set_index(['x','y'])

df_2 = df_2.join(df_0, lsuffix = '_old', rsuffix = '_new', how='outer')



#df_0.reset_index().pivot('x','y','classes')



































