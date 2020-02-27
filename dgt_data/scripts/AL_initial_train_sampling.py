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
PATH = r'C:\Users\arman\Desktop\ActiveLearning\Experiment\my_data'

s1_path = join(PATH, 'sentinel1')
s2_path = join(PATH, 'sentinel2')
buffered_cos_path = join(PATH, 'data')
shape_path = join(PATH,  'cos_shape', 'COS_2015_clipped.shp')

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
for path in (buffered_cos_path, s2_path, s1_path):
    querie = join(path, search_criteria)   
    raster_path = glob.glob(querie)
    rasters.extend(raster_path)
# choose subset of the rasters
rasters =  [rasters[index] for index in [1,2,3]]


# read rasters as array, fill missing values and appent to the list
imputer = SimpleImputer(missing_values = np.nan, strategy ="mean") 
bands_list = []

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
del bands_list

_, _, n_bands = data_stack.shape
# number of pixels on one band
n_pixels = rows*cols
# stack bands as columns each (2D)
flat_pixels = data_stack.reshape((n_pixels, n_bands))
del data_stack
# drop Nones
index = np.argwhere(flat_pixels[:,0]==-1)
all_stack = np.delete(flat_pixels, index, axis = 0)
del flat_pixels
del index
#Counter(all_stack[:,0])

labels_row = all_stack[:,0]

#==============================
# Sampling the data
#==============================
n_samples = 500
sampled_data = []

for label in list(np.unique(labels_row)):
    index = np.argwhere(labels_row == label).reshape(-1)
    choice = np.random.choice(index, size=n_samples, replace=False)
    sampled = all_stack[choice, :]
    sampled_data.append(sampled)

# make stack of selected sample darrays
sampled_stack = np.vstack(sampled_data)
del all_stack
del sampled_data

# give the band names 
columns = ['classes'] + \
          BANDS_NAME['S2_N1'] +\
          BANDS_NAME['S2_N2'] +\
          ['x','y']
          #BANDS_NAME['S1']+\
          #list(map(lambda x: (x+'3'), BANDS_NAME['S1_GLCM']))+\
     
# Make a DataFrame from the long_stacked images
df_coords = pd.DataFrame(sampled_stack, columns=columns)
del sampled_stack

#==============================
# Pass the real CRS to X and Y
#==============================
# set the transformation factor as Affine
transf = src.meta['transform']

# select the centeroid of the pixel.
# for this we reduce the original pixel coordinates by
# the half of the pixel size (5 m in this case)
transf = Affine(10.0, 0.0, 499985.0,
               0.0, -10.0, 4400045.0)

geo_coords = df_coords.apply(
    lambda row: transf*(row['x'], row['y']), axis=1
)


df_coords['x_geo'] = geo_coords.apply(lambda x: x[0])
df_coords['y_geo'] = geo_coords.apply(lambda x: x[1])

# drop pixel coords for raster
df_coords = df_coords.drop(columns=['x','y'])
# save data to csv
df_coords.to_csv(join(out_path,"train_sampled.csv"), sep=',',header=True, index=True)

df_to_shape = df_coords.loc[:,['classes','x_geo','y_geo']]

def pd_to_gpd(df, longitude_col='x_geo', latitude_col='y_geo'):
    """
    Converts a pandas dataframe to geopandas.
    Params:
    - df: Pandas dataframe to convert
    - longitude_col: column containing the longitude coordinates
    - latitude_col: column containing the latitude coordinates
    Returns a geopandas dataframe.
    """
    geom_col = [Point(xy) for xy in zip(df[longitude_col], df[latitude_col])]
    df = df.drop([longitude_col, latitude_col], axis=1)
    crs = {'init': 'epsg:32629'}
    df = gpd.GeoDataFrame(df, crs=crs, geometry=geom_col)
    return df

# make a GeoDataFram from pd DataFrame
gpd_point = pd_to_gpd(df_to_shape, longitude_col='x_geo', latitude_col='y_geo')

# save the shapefile
gpd_point.to_file(join(out_path,"train_sampled.shp"))

'''
#==============================
# Image classification
#==============================



# reshape the vector raster to its original state to form the raster
arr_to_rast =np.reshape(stack, (rows, cols))

# Output the simulated landcover raster
def write_geotiff(dest_path, data, geo_transform, projection):
    #"""Create a GeoTIFF file with the given data."""
    driver= gdal.GetDriverByName('GTiff')
    rows, cols= data.shape
    dataset= driver.Create(fname, cols, rows, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band= dataset.GetRasterBand(1)
    band.WriteArray(data)
    dataset=None
    
# write and close the file for 2028 Business as usual
write_geotiff(dest_path, arr_to_rast, transform, proj)

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
