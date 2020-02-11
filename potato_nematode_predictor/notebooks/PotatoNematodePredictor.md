# Potato Nematode Predictor
This work contains the public sector consultancy work on a potato nematode predictor carried out by Aarhus University.

Start by configuring the notebook:

```python
import wget
import geopandas
import os
import rasterio
import sys
import xarray
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from zipfile import ZipFile
from tqdm.autonotebook import tqdm

from utils import RasterstatsMultiProc

# Automatically prints execution time for the individual cells
%load_ext autotime

# Automatically reloads functions defined in external files
%load_ext autoreload
%autoreload 2

# The path to the project (so absoute file paths can be used throughout the notebook)
PROJ_PATH = Path.cwd().parent

# Define which field polygons should be used for analysis (2017 to 2019 seem to follow the same metadata format)
FIELD_POLYGONS = ['FieldPolygons2017', 'FieldPolygons2018', 'FieldPolygons2019']

# Define global flags
MULTI_PROC_ZONAL_STATS = False
ALL_TOUCHED = False
```

---
Download the field polygons from The Danish Agricultural Agency:

```python
# Downloaded files will go into the 'data/external' folder
dest_folder = PROJ_PATH / 'data' / 'external'
if not dest_folder.exists():
    os.makedirs(dest_folder)
    
# Define the download links for the field polygons for the individual years
file_url_mapping = {
    'FieldPolygons2016.zip': 'https://kortdata.fvm.dk/download/DownloadStream?id=3037da0f2744a85adc8b08ca5c31c3cb',
    'FieldPolygons2017.zip': 'https://kortdata.fvm.dk/download/DownloadStream?id=d0c8946763e465bf9f6160a6bc40531f',
    'FieldPolygons2018.zip': 'https://kortdata.fvm.dk/download/DownloadStream?id=cfb1b47130b7276f8515fbaae60bde2a',
    'FieldPolygons2019.zip': 'https://kortdata.fvm.dk/download/DownloadStream?id=3d19613ac986ed05a7c301319738e332'
}

# Download the zipfiles
for filename, url in file_url_mapping.items():
    dest_path = PROJ_PATH / 'data' / 'external' / filename
    if not dest_path.exists():
        wget.download(url, str(dest_path))
        print("File has been downloaded: " + filename)
    else:
        print("File already exists: " + str(PROJ_PATH / 'data' / 'external' / filename))
```

---
Then extract the zipfiles:

```python
# The extracted zipfiles will go into the 'data/raw' folder
for zipfile in (PROJ_PATH / 'data' / 'external').glob('**/*.zip'):
    dest_folder = PROJ_PATH / 'data' / 'raw' / zipfile.stem   
    if not dest_folder.exists():
        with ZipFile(str(zipfile), 'r') as zipObj:
            zipObj.extractall(str(dest_folder))
        print("Zipfile has been extracted: " + str(zipfile))
    else:
        print("Zipfile has already been extracted: " + str(zipfile))
```

---
Now load the shapefiles into geopandas dataframes:

```python
def load_shp(shp_name):
    # Load shapefile into dataframe and remove NaN rows
    shp_file_path = list((PROJ_PATH / 'data' / 'raw' / shp_name).glob('**/*.shp'))[0]
    df = geopandas.read_file(str(shp_file_path))
    df = df.dropna()
    
    # Change all column names to be lower-case to make the naming consistent across years (https://stackoverflow.com/a/36362607/12045808)
    df.columns = map(str.lower, df.columns)
    
    return df

# Load the dataframes into a dict, with each year as a key
df_all = {}
for df_name in FIELD_POLYGONS:
    df = load_shp(df_name)
    df_all[df_name] = df
```

---
Find the potato fields and count the number of unique sorts:

```python
def extract_potato_fields(df):
    # Create a new dataframe with all the different types of potatoes
    df = df[df['afgroede'].str.contains("kartof", case=False)]  

    # Find the different potato types, count the number of fields for each type, and calculate total area for each type
    for potato_type in sorted(df['afgroede'].unique()):
        num_fields = df[df['afgroede'] == potato_type].shape[0]
        sum_area = df[df['afgroede'] == potato_type]['imk_areal'].sum()
        print("There are " + str(num_fields) + " fields (total area = " + str(int(sum_area)) + " ha) of type: " + potato_type)
        
    return df 

# Extract the potato fields and load them into a new dict with each year as a key
df_potato = {}
for df_name, df in df_all.items():
    print("### Analyzing " + df_name + " ###")
    df_potato[df_name] = extract_potato_fields(df)
    print("")
```

---
Calculate zonal statistics for the the potato fields for the different radar data measurements:

```python
tif = list((PROJ_PATH / 'data' / 'raw' / 'Sentinel-1').glob('*.tif'))[0]
with rasterio.open(tif) as src:
    tif_crs = src.crs
    print("Projection used is: " + str(tif_crs))

for df_name, df in df_potato.items():
    # Set the CRS in the geodataframe to be wkt format (otherwise you won't be able to save as a shapefile)
    df_potato[df_name] = df_potato[df_name].to_crs({'init': tif_crs})
```

```python
tifs = sorted((PROJ_PATH / 'data' / 'raw' / 'Sentinel-1').glob('*.tif'))
df_potato_stats = df_potato.copy()

for df_name, df in df_potato.items(): # Loop over all field polygon years
    pkl_name = df_name + '_stats' 
    pkl_path = (PROJ_PATH / 'data' / 'processed' / pkl_name).with_suffix('.pkl')
    if pkl_path.exists():
        print("Zonal statistics have already been calculated for: " + df_name)
    else:
        print("Calculating zonal statistics for: " + df_name)
        #df = df.head(20)  # For debugging to (ie. only process 20 fields)
        for tif in tqdm(tifs):  # Loop over all Sentinel-1 images
            for band in range(1, 4):  # Loop over all three bands (indexed 1 to 3)
                rasterstatsmulti = RasterstatsMultiProc(df=df, tif=tif, all_touched=ALL_TOUCHED)

                if MULTI_PROC_ZONAL_STATS:
                    results_df = rasterstatsmulti.calc_zonal_stats_multiproc()     
                else:
                    results_df = rasterstatsmulti.calc_zonal_stats(band=band, prog_bar=False) 

                del rasterstatsmulti

                stats_cols = {
                    'min': tif.stem + '_B' + str(band) + '_min',
                    'max': tif.stem + '_B' + str(band) + '_max',
                    'mean': tif.stem + '_B' + str(band) + '_mean',
                    'std': tif.stem + '_B' + str(band) + '_std',
                    'median': tif.stem + '_B' + str(band) + '_median',
                }

                results_df = results_df.rename(columns=stats_cols)

                # Note: The * operator iterates through the list (https://stackoverflow.com/a/56736691/12045808)
                df_potato_stats[df_name] = df_potato_stats[df_name].merge(results_df[['id', *stats_cols.values()]], left_on='id', right_on='id')

        if not shp_path.parent.exists():
            os.makedirs(shp_path.parent)

        # Set the CRS in the geodataframe to be wkt format (otherwise you won't be able to save as a shapefile)
        #df_potato_stats[df_name].crs = df_potato_stats[df_name].crs['init'].to_wkt()
        #df_potato_stats[df_name] = df_potato_stats[df_name].dropna()
        df_potato_stats[df_name].to_pickle(pkl_path) 
```

```python
# We now want to create an xarray dataset based on the dataframe, with the zonal statistics as extra dimensions
tifs = sorted((PROJ_PATH / 'data' / 'raw' / 'Sentinel-1').glob('*.tif'))
df_potato_stats = df_potato.copy()

for df_name, df in df_potato.items(): # Loop over all field polygon years
    pkl_name = df_name + '_stats_multiindex' 
    pkl_path = (PROJ_PATH / 'data' / 'processed' / pkl_name).with_suffix('.pkl')
    #if pkl_path.exists():
    if False:
        print("Zonal statistics have already been calculated for: " + df_name)
    else:
        print("Calculating zonal statistics for: " + df_name)
        df = df.head(20)  # For debugging (ie. only process 20 fields)
        df = df.set_index('id')
        ds = df.to_xarray()
        '''
        #for tif in tqdm(tifs):  # Loop over all Sentinel-1 images
        for tif in tqdm(tifs[0:4]):  # Loop over all Sentinel-1 images
            # Get metadata for satellite pass from the filename of the .tif file
            satellite = tif.stem[0:3]
            date = tif.stem[4:12]
            pass_mode = tif.stem[20:23]
            relative_orbit = tif.stem[24:27]
            
            # Perform zonal statistics on all bands
            for band in range(1, 4):  # Loop over all three bands (indexed 1 to 3)
                rasterstatsmulti = RasterstatsMultiProc(df=df, tif=tif, all_touched=ALL_TOUCHED)

                if MULTI_PROC_ZONAL_STATS:
                    results_df = rasterstatsmulti.calc_zonal_stats_multiproc()     
                else:
                    results_df = rasterstatsmulti.calc_zonal_stats(band=band, prog_bar=False) 

                del rasterstatsmulti

                stats_cols = {
                    'min': tif.stem + '_B' + str(band) + '_min',
                    'max': tif.stem + '_B' + str(band) + '_max',
                    'mean': tif.stem + '_B' + str(band) + '_mean',
                    'std': tif.stem + '_B' + str(band) + '_std',
                    'median': tif.stem + '_B' + str(band) + '_median',
                }

                results_df = results_df.rename(columns=stats_cols)

                # Note: The * operator iterates through the list (https://stackoverflow.com/a/56736691/12045808)
                df_potato_stats[df_name] = df_potato_stats[df_name].merge(results_df[['id', *stats_cols.values()]], left_on='id', right_on='id')
        '''

        if not pkl_path.parent.exists():
            os.makedirs(pkl_path.parent)

        # Set the CRS in the geodataframe to be wkt format (otherwise you won't be able to save as a shapefile)
        #df_potato_stats[df_name].crs = df_potato_stats[df_name].crs['init'].to_wkt()
        #df_potato_stats[df_name] = df_potato_stats[df_name].dropna()
        df_potato_stats[df_name].to_pickle(pkl_path) 
    break
```

```python
ds
```

```python
print(ds['imk_areal'][2].values)
```

```python
ds.expand_dims('stats')
ds['stats'] = 5
print(ds['stats'].values)
```

```python
df_potato_stats['FieldPolygons2017'].head(5)
```

```python
df_potato_stats = {}
for df_name in FIELD_POLYGONS:
    pkl_name = df_name + '_stats'
    pkl_path = (PROJ_PATH / 'data' / 'processed' / pkl_name).with_suffix('.pkl')
    df = pd.read_pickle(pkl_path)
    df_potato_stats[df_name] = df
    

x = []
for column in df_potato_stats['FieldPolygons2017']:
    if 'S1A' in column and 'B1_mean' in column:
        value = df_potato_stats['FieldPolygons2017'][column].loc[8]
        x.append(value)
    #print(df_potato_stats[column])
    
print(x)
plt.plot(x)
```
