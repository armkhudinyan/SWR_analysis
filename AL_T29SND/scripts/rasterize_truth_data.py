# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:52:05 2020

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

import optparse
optparser = optparse.OptionParser()
optparser.add_option(
    "--n_run",
    help = "Mandatory parameter. Iteration number."
)

options = optparser.parse_args()[0]
if options.n_run is None:   # if filename is not given
    optparser.error('Mandatory argument n_run not given.')

n_run = int(options.n_run)

#===================
# Defining paths
#===================
#sys.path.append(join(dirname(__file__), '..', '..'))
#PATH = r'C:\Users\arman\Documents\GitHub\active-learning\AL_T29SND'

PATH = join(dirname(__file__),'..')
TRAIN_PATH = join(PATH, 'train_data','train_best_40.csv')
FEATURE_NAME_PATH = join(PATH, 'feature_importance', 'feature_rankings.csv')


SHAPE_PATH = join(PATH,  'truth_patches', f'truth_patches_{n_run}.shp')

OUT_PATH = join(PATH, 'truth_rasters', f'truth_patches_{n_run}.tif')

#==============================
# rasterize shapefiles
#==============================
# get shapefiles
truth_patches = gpd.read_file(SHAPE_PATH)
#gdf_cosTejo = gpd.read_file(path_Laziria_Tejo)

# get a raster from the proper tile for the extents and metadata
search_criteria = "*.tif"
querie = join(METRICS_PATH, search_criteria)
raster_path = glob.glob(querie)

src = rasterio.open(raster_path[0])
proj = src.crs
out_meta = src.meta.copy()

# First of all check the coordinate reference systems of the files
print('truth_patches proj:', truth_patches.crs)
print('raster proj:', src.crs)
# In case of missmatching, the layers should be transformed into one crs
#truth_patches = truth_patches.to_crs(proj)


'''
#========================
# If we want to clip shape with shape before rasterization
#========================
# save thr shapefile with new coordinates
#truth_patches.to_file(join(PATH, 'cos','truth_patches_transf.shp'))
# another way is to do with fiona library => fiona.transform.transform_geom

ras_intersection = gpd.overlay(truth_patches, some_other_shape, how='intersection')

# vizualize the reuslts
ax = ras_intersection.plot(cmap='tab10')
truth_patches.plot(ax=ax, facecolor='none', edgecolor='k');
some_other_shape.plot(ax=ax, facecolor='none', edgecolor='k');

megaclass = Counter(truth_patches['Megaclasse'])
df_megaclass = pd.Series(dict(Counter(megaclass)))
'''

# print info on shapefile
print(truth_patches.info())
print(f'\nMajor Classes: \n{truth_patches.groupby("Class").size()}')

#truth_patches.plot()
# get the bouns
#bounds = gdf_cos15.bounds.iloc[0]
#crs = gdf_cos15.crs


# rasterize COS
#labels = {v:i for i, v in enumerate(truth_patches['rstr_cd'].unique())}

shapes = [
    (geom, int(value))
    for geom, value
    in zip(truth_patches['geometry'], truth_patches['rstr_cd'])#.map(labels))
]

#rasterize using the shape and transform of the satellite image
out_shape = src.shape
transform = src.transform
#out_meta  = src.meta
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
                   "height": src.height,
                   "width": src.width,
                   "transform": transform,
                   "count": 1,
        			 # Specify to any crs by defining here. if you don't do so,
        			 # it sets up to wgs84 geographic by default (weird tho)
                   #"crs": "+proj=utm +zone=29 +ellps=WGS84 +datum=WGS84 +units=m +no_defs "
                   "crs":  out_meta['crs']
                  }

# Write the raster to disk
with rasterio.open(OUT_PATH, "w", **out_meta) as dest:
    dest.write(array_to_rast, indexes=1)
