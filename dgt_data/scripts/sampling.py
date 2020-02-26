from rasterio import features
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from collections import Counter
from shapely.geometry import Point


gdf = gpd.read_file('COS2018_ISIP_ALL_31C_newPerma_noBurn_NDVI.shp')

labels = {v:i for i, v in enumerate(gdf['CLASS'].unique())}
shapes = [(geom, value) for geom, value in zip(gdf['geometry'], gdf['CLASS'].map(labels))]

# get original class frequencies

bounds = gdf.bounds
out_shape = (
    int(np.ceil((bounds.maxy.max()-bounds.miny.min())/10)),
    int(np.ceil((bounds.maxx.max()-bounds.minx.min())/10))
)
transf = Affine(10.0, 0.0, bounds.minx.min(),
         0.0, -10.0, bounds.maxy.max())

array = features.rasterize(
    shapes=shapes,
    out_shape=out_shape,
    fill=-1,
    transform=transf,
    all_touched=False,
    default_value=1,
    dtype='int32',
)

freq = pd.Series(Counter(array.flatten())).drop(index=-1)
freq_perc = freq/freq.sum()

# Get point coordinates
gdf.geometry = gdf.buffer(-5)
gdf = gdf[~gdf.is_empty]

bounds = gdf.bounds
out_shape = (
    int(np.ceil((bounds.maxy.max()-bounds.miny.min())/10)),
    int(np.ceil((bounds.maxx.max()-bounds.minx.min())/10))
)
transf = Affine(10.0, 0.0, bounds.minx.min(),
         0.0, -10.0, bounds.maxy.max())

array = features.rasterize(
    shapes=shapes,
    out_shape=out_shape,
    fill=-1,
    transform=transf,
    all_touched=False,
    default_value=1,
    dtype='int32',
)

columns = ['y', 'x']+['class']
shp  = array.shape
coords = np.indices(shp)
data = np.concatenate([
    coords,
    np.expand_dims(array, 0)],
    axis=0
).reshape((len(columns), shp[0]*shp[1]))
df_coords = pd.DataFrame(data=data.T, columns=columns)

df_coords = df_coords[df_coords['class']!=-1]
sample_size = (freq_perc*1_000_000).astype(int)
df_coords['class1'] = df_coords['class']
df_coords_sampled = df_coords.groupby('class')\
    .apply(lambda x: x.sample(sample_size[x['class1'].iloc[0]]))\
    .drop(columns='class1')\
    .reset_index(drop=True)

final_coords = df_coords_sampled.apply(
    lambda row: transf*(row['x'], row['y']), axis=1
)

df_coords_sampled['x_geo'] = final_coords.apply(lambda x: x[0])
df_coords_sampled['y_geo'] = final_coords.apply(lambda x: x[1])

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

gdf.plot('CLASS')
pd_to_gpd(df_coords_sampled).plot()

labels_rev = {i:v for v, i in labels.items()}

df_coords_sampled['class_decoded'] = df_coords_sampled['class'].map(labels_rev)
df_coords_sampled.to_csv('sampled_point_coordinates.csv', index=False)
