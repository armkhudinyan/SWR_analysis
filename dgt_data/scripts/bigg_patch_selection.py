# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 23:50:08 2020

@author: arman
"""
from os.path import join, dirname
import pandas as pd
import geopandas as gpd

shp_path = r'C:\Users\arman\Desktop\ActiveLearning\Experiment\dgt_T29SND\delete'
gdf_uncert = gpd.read_file(join(shp_path,'uncert_intersect.shp'))

gdf_uncert.groupby("Nomenclatu").size()
gdf_uncert.groupby("classes").size()

classes = gdf_uncert.classes.unique().tolist()

class_id = gdf_uncert.classes.unique()
class_size = gdf_uncert.groupby('classes').size()
class_size_dict = dict(class_size)

to_list = [class_size_dict[keys] for keys in class_size_dict]  

# create list containing sample size per class
sample_size = []

for i in to_list:
    if i >10:
        sample_size.append(10)
    else:
        sample_size.append(i)

#df_sample_size = pd.Series(sample_size)

df = pd.DataFrame({'classes':classes, 'size':sample_size})

# sample data by given sample size
largest_patches = gdf_uncert.groupby('classes')\
               .apply(lambda x: x.sort_values(by='Area', ascending=False)\
               .iloc[:df.loc[df['classes']== x['classes'].iloc[0], 'size'].iloc[0],:])\
               .reset_index(drop=True)
    
    # save the shapefile
largest_patches.to_file('uncert_10_pathches.shp')





