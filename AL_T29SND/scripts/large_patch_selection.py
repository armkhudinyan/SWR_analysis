# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 23:50:08 2020

@author: arman
"""
import os
from os.path import join, dirname
import pandas as pd
import geopandas as gpd



import optparse
optparser = optparse.OptionParser()
optparser.add_option(
    "--n_run",
    help = "Mandatory parameter. Iteration number."
)

optparser.add_option(
    "--n_patches", default = '10',
    help = "Number of patches per class to output in the final uncertainty patches passed to photointerpreters."
)

optparser.add_option(
    "--uncertainty_path",
    help = "Path to the post-processed uncertainty map shapefile."
)

options = optparser.parse_args()[0]
if options.n_run is None:   # if filename is not given
    optparser.error('Mandatory argument n_run not given.')
if options.gdf_uncert is None:   # if filename is not given
    optparser.error('Mandatory argument gdf_uncert not given.')

n_run = int(options.n_run)
n_patches = int(options.n_patches)
uncertainty_path = options.uncertainty_path

gdf_uncert = gpd.read_file(uncertainty_path)

PATH = join(dirname(__file__), '..', 'uncertainty_patches')

#==============================
# Make dataframe containing class names and class IDs
#==============================
print('class sizes' ,gdf_uncert.groupby("Nomenclatu").size())
# Recalculate polygon areas 
gdf_uncert['Area'] = gdf_uncert.geometry.area

class_id = sorted(gdf_uncert.classes.unique().tolist()) # should be sorted to match with class_size order
class_size = gdf_uncert.groupby('classes').size()
class_size_dict = dict(class_size)

to_list = [class_size_dict[keys] for keys in class_size_dict]  

# create list containing sample size per class
sample_size = []

for i in to_list:
    if i > n_patches:
        sample_size.append(n_patches)
    else:
        sample_size.append(i)

df = pd.DataFrame({'classes':class_id, 'size':sample_size})

#==============================
# large patches selection
#==============================
# sample data by given sample size
largest_patches = gdf_uncert.groupby('classes')\
               .apply(lambda x: x.sort_values(by='Area', ascending=False)\
               .iloc[:df.loc[df['classes']== x['classes'].iloc[0], 'size'].iloc[0],:])\
               .reset_index(drop=True)

# drop classes "Irrigated" and "Rainfed" (-1 and -2)
largest_patches = largest_patches[~largest_patches['classes'].isin([-1,-2])]
print('class sizes' ,largest_patches.groupby("Nomenclatu").size())

# specify the directory to save the shapefile
largest_patches.to_file(join(PATH, f'uncert_{n_patches}_patches_{n_run}.shp'))
