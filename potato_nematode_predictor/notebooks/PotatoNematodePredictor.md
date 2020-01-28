# Potato Nematode Predictor
This work contains the public sector consultancy work on a potato nematode predictor carried out by Aarhus University.

Start by configuring the notebook:

```python
import wget
import geopandas
import qgrid
import os

from pathlib import Path
from zipfile import ZipFile

%load_ext autotime
%load_ext autoreload
%autoreload 2

PROJ_PATH = Path.cwd().parent
FIELD_POLYGONS = ['FieldPolygons2016', 'FieldPolygons2017', 'FieldPolygons2018', 'FieldPolygons2019']
```

Download the field polygons from The Danish Agricultural Agency:

```python
dest_folder = PROJ_PATH / 'data' / 'external'
if not dest_folder.exists():
    os.makedirs(dest_folder)
    
file_url_mapping = {
    'FieldPolygons2016.zip': 'https://kortdata.fvm.dk/download/DownloadStream?id=3037da0f2744a85adc8b08ca5c31c3cb',
    'FieldPolygons2017.zip': 'https://kortdata.fvm.dk/download/DownloadStream?id=d0c8946763e465bf9f6160a6bc40531f',
    'FieldPolygons2018.zip': 'https://kortdata.fvm.dk/download/DownloadStream?id=cfb1b47130b7276f8515fbaae60bde2a',
    'FieldPolygons2019.zip': 'https://kortdata.fvm.dk/download/DownloadStream?id=3d19613ac986ed05a7c301319738e332'
}

for filename, url in file_url_mapping.items():
    dest_path = PROJ_PATH / 'data' / 'external' / filename
    if not dest_path.exists():
        wget.download(url, str(dest_path))
        print("File has been downloaded: " + filename)
    else:
        print("File already exists: " + str(PROJ_PATH / 'data' / 'external' / filename))
```

Then extract the zipfiles:

```python
for zipfile in (PROJ_PATH / 'data' / 'external').glob('**/*.zip'):
    dest_folder = PROJ_PATH / 'data' / 'raw' / zipfile.stem
    if not dest_folder.exists():
        with ZipFile(str(zipfile), 'r') as zipObj:
            zipObj.extractall(str(dest_folder))
        print("Zipfile has been extracted: " + str(zipfile))
    else:
        print("Zipfile has already been extracted: " + str(zipfile))
```

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
df_fields = {}
for df_name in FIELD_POLYGONS:
    df = load_shp(df_name)
    df_fields[df_name] = df
```

Find the potato fields and count the number of unique sorts:

```python
def count_potato_fields(df):
    # Create a new dataframe with all the different types of potatoes
    df = df[df['afgroede'].str.contains("kartof", case=False)]  

    # Find the different potato types and count the number of fields for each type
    for crop_type in df['afgroede'].unique():
        num_fields = df[df['afgroede'] == crop_type].shape[0]
        print("There are " + str(num_fields)  + " fields of crop type: " + crop_type)

for df_name, df in df_fields.items():
    print("### Showing results for: " + df_name + " ###")
    count_potato_fields(df)
    print("")
```

```python

```
