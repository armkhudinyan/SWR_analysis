# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 23:50:08 2020

@author: arman
"""
import os
from os.path import join
import pandas as pd
import geopandas as gpd

PATH = r'C:\Users\arman\Desktop\ActiveLearning\Experiment\dgt_T29SND\delete'
gdf_uncert = gpd.read_file(join(PATH,'uncert_intersect.shp'))

# define the amount of patches to be selected per class
n_patches = 10

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
os.chdir(PATH)
largest_patches.to_file(f'uncert_{n_patches}_pathches.shp')




























