# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 23:50:08 2020

@author: arman
"""
import os
from os.path import join, dirname
import pandas as pd
import geopandas as gpd

shp_path = r'C:\Users\arman\Desktop\ActiveLearning\Experiment\dgt_T29SND\delete'
gdf_uncert = gpd.read_file(join(shp_path,'uncert_intersect.shp'))

# define the amount of patches to be selected per class
patch_num = 10

#==============================
# Make dataframe containing class names and class IDs
#==============================
print('class sizes' ,gdf_uncert.groupby("Nomenclatu").size())
#gdf_uncert.groupby("classes").size()

class_id = sorted(gdf_uncert.classes.unique().tolist()) # should be sorted to match with class_size order
class_size = gdf_uncert.groupby('classes').size()
class_size_dict = dict(class_size)

to_list = [class_size_dict[keys] for keys in class_size_dict]  

# create list containing sample size per class
sample_size = []

for i in to_list:
    if i > patch_num:
        sample_size.append(patch_num)
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
    
# specify the directory to save save the shapefile
os.chdir(r'C:\Users\arman\Desktop\ActiveLearning\Experiment\dgt_T29SND\delete')
largest_patches.to_file('uncert_10_pathches2.shp')




