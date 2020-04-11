# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 20:51:21 2020

@author: arman
"""

import time
import os
import sys
from os.path import join, dirname

import numpy as np
import pandas as pd

import rasterio
from rasterio.merge import merge
from rasterio.plot import show
from rasterio import features
import shapely
import geopandas as gpd
import gdal
import ogr

from sklearn.impute import SimpleImputer 
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from collections import Counter

#===================
# Defining paths
#===================
#sys.path.append(join(dirname(__file__), '..', '..'))
PATH = r'C:\Users\arman\Desktop\ActiveLearning\Experiment\my_data'

s1_path = join(PATH, 'sentinel1')
s2_path = join(PATH, 'sentinel2')
buffered_cos_path = join(PATH, 'data', 'buffered20_cos.tif')
shape_path = join(PATH,  'cos_shape', 'COS_2015_clipped.shp')

out_path = join(PATH, 'data')


# Bands name for the GEE exported rasters
#BANDS_NAME_S2_N1   = ['B2',  'B3',  'B4',   'B5',   'B8'  ]
#BANDS_NAME_S2_N2   = ['B8A', 'B11', 'NDVI', 'NDMI', 'NDBI']
#BANDS_NAME_S1      = ['fallVV', 'winterVV', 'springVV', 'summerVV', 
#                      'fallVH', 'winterVH', 'springVH', 'summerVH']
#BANDS_NAME_S1_GLCM = ['savg', 'ent', 'corr', 'idm', 'var', 'diss', 'contrast', 'asm']

BANDS_NAME = {
            'BANDS_NAME_S2_N1' :  ['B2',  'B3',  'B4',   'B5',   'B8'  ],
            'BANDS_NAME_S2_N2' :  ['B8A', 'B11', 'NDVI', 'NDMI', 'NDBI'],
            'BANDS_NAME_S1'    :  ['fallVV', 'winterVV', 'springVV', 'summerVV', 
                                  'fallVH', 'winterVH', 'springVH', 'summerVH'],
            'BANDS_NAME_S1_GLCM': ['savg', 'ent', 'corr', 'idm', 'var', 'diss', 'contrast', 'asm']
            }

#==============================
# rasterize shapefiles
#==============================
# get shapefiles
gdf_cos15 = gpd.read_file(shape_path)
#gdf_cosTejo = gpd.read_file(path_Laziria_Tejo)

# get a .tif raster fot the extents
src = rasterio.open(join(s1_path,os.listdir(s1_path)[0]))
proj = src.crs
out_meta = src.meta.copy()

# First of all check the coordinate reference systems of the files 
print('gdf_cos15 proj:', gdf_cos15.crs)
#print('gdf_cosTejo proj:', gdf_cosTejo.crs)
print('raster proj:', src.crs)

# In case of missmatching, the layers should be transformed into one crs
#gdf_cosTejo_transf = gdf_cosTejo.to_crs(proj)

'''
If we want to clip shape with shape before rasterization
# save thr shapefile with new coordinates
#gdf_cosTejo_transf.to_file(join(PATH, 'cos','cos_lazirio_tejo.shp'))
# another way is to do with fiona library => fiona.transform.transform_geom

res_intersection = gpd.overlay(gdf_cosTejo_transf, gdf_cos15, how='intersection')

# vizualize the reuslts
ax = res_intersection.plot(cmap='tab10')
gdf_cosTejo_transf.plot(ax=ax, facecolor='none', edgecolor='k');
gdf_cos15.plot(ax=ax, facecolor='none', edgecolor='k');

megaclass = Counter(gdf_cosTejo_transf['Megaclasse'])
df_megaclass = pd.Series(dict(Counter(megaclass)))
'''

# print infor on shapefile
#print(gdf_cos15.info())
#print('CRS:',gdf_cos15.crs)
print(f'\nMajor Classes: \n{gdf_cos15.groupby("Legend").size()}')
#print(f'\nMajor Classes: \n{df_megaclass}')

#gdf_cos15.plot()
# get the bouns
#bounds = gdf_cos15.bounds.iloc[0]
#crs = gdf_cos15.crs


# rasterize COS
labels = {v:i for i, v in enumerate(gdf_cos15['Legend'].unique())}

shapes = [
    (geom, value)
    for geom, value
    in zip(gdf_cos15['geometry'], gdf_cos15['Legend'].map(labels))
]


'''
# if we keep the shapefile extent as an exported tiff extent
out_shape = (
    int(np.ceil((bounds.maxy-bounds.miny)/10)),
    int(np.ceil((bounds.maxx-bounds.minx)/10))
)

transf = Affine(10.0, 0.0, bounds.minx,
         0.0, -10.0, bounds.maxy)

raster = features.rasterize(
    shapes=shapes,
    out_shape=out_shape,
    fill=-1,
    transform=transf,
    all_touched=False,
    default_value=1,
    dtype='int32',
)
'''

#rasterize using the shape and transform of the satellite image
out_shape = src.shape
transform = src.transform
out_meta  = src.meta
# 'shapes' in rasterio.rasterize  is iterable of (geometry, value) pairs 
# or iterable over

array_to_rast = features.rasterize(
                                   shapes=shapes, 
                                   out_shape=out_shape, 
                                   transform=transform, 
                                   fill=-1,
                                   all_touched=False,
                                   #default_value=1,
                                   dtype='float32',
                                   )

out_meta= {"driver": "GTiff",
                   "dtype": 'float32',
                   "nodata": None,
                   #"height": src.height,
                   #"width": src.width,
                   "transform": transform,
                   "count": 1,
        			 # Specify to any crs by defining here. if you don't do so, 
        			 # it sets up to wgs84 geographic by default (weird tho)
                   #"crs": "+proj=utm +zone=29 +ellps=WGS84 +datum=WGS84 +units=m +no_defs "
                   "crs":  out_meta['crs']
                  }
                  
# Write the raster to disk
with rasterio.open(join(out_path, 'rasterized_cosTejo.tif'), "w", **out_meta) as dest:
    dest.write(array_to_rast, indexes=1)  
    
# apply negative buffer and repeat rasterization process
freq = pd.Series(Counter(array_to_rast.flatten())).drop(index=-1)
freq_perc = freq/freq.sum()

#==============================
# Apply negative buffer
#==============================
gdf_cos15.geometry = gdf_cos15.buffer(-20)
gdf_cos15 = gdf_cos15[~gdf_cos15.is_empty]


# rasterize COS
labels = {v:i for i, v in enumerate(gdf_cos15['Legend'].unique())}

shapes = [
    (geom, value)
    for geom, value
    in zip(gdf_cos15['geometry'], gdf_cos15['Legend'].map(labels))
    ]

# create array to be written as raster
buffered = features.rasterize( shapes=shapes, 
                               out_shape=out_shape, 
                               transform=transform, 
                               fill=-1,
                               all_touched=False,
                               #default_value=1,
                               dtype='float32')

# define the raster parameters to be written
out_meta.update({  "driver": "GTiff",
                   "dtype": 'float32',
                   #"nodata": None,
                   "height": src.height,
                   "width": src.width,
                   "transform": transform,
                   "count": 1,
                   # to any crs by defining here. if you don't do so, 
                   # up to wgs84 geographic by default (weird tho)
                   #"crs": "+proj=utm +zone=29 +ellps=WGS84 +datum=WGS84 +units=m +no_defs "
                   "crs":  out_meta['crs']
                  })

with rasterio.open(join(out_path, 'buffer_1_cos2015.tif'), "w", **out_meta) as dest:
    dest.write(buffered,indexes=1)  










