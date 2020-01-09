# Potato Nematode Predictor
This work contains the public sector consultancy work on a potato nematode predictor carried out by Aarhus University.

Start by configuring the notebook:

```python
import wget
import geopandas
import qgrid

from pathlib import Path
from zipfile import ZipFile

%load_ext autotime
%load_ext autoreload
%autoreload 2

proj_path = Path.cwd().parent
```

Download the field polygons from The Danish Agricultural Agency:

```python
dest_folder = proj_path / 'data' / 'external'
file_url_mapping = {
    'FieldPolygons2016.zip': 'http://i3.ytimg.com/vi/J---aiyznGQ/mqdefault.jpg'
}

for filename, url in file_url_mapping.items():
    dest_file = proj_path / 'data' / 'external' / filename
    if not dest_file.exists():
        wget.download(url, str())
        print("File has been downloaded: " + filename)
    else:
        print("File already exists: " + str(proj_path / 'data' / 'external' / filename))
```

Then extract the zipfiles:

```python
for zipfile in (proj_path / 'data' / 'external').glob('**/*.zip'):
    dest_folder = proj_path / 'data' / 'raw' / zipfile.stem
    if not dest_folder.exists():
        with ZipFile(str(zipfile), 'r') as zipObj:
            zipObj.extractall(str(dest_folder))
        print("Zipfile has been extracted: " + str(zipfile))
    else:
        print("Zipfile has already been extracted: " + str(zipfile))
```

Now load the shapefiles into geopandas dataframes and look at 3 example fields:

```python
df_fields_2016 = geopandas.read_file(str(proj_path / 'data' / 'raw' / 'FieldPolygons2016' / 'Marker_2016_CVR.shp'))
df_fields_2016 = df_fields_2016.dropna()
df_fields_2016.head(3)
```

Find the potato fields and count the number of unique sorts:

```python
df_potato_2016 = df_fields_2016[df_fields_2016['Afgroede'].str.contains("kartof", case=False)]  

for crop_type in df_potato_2016['Afgroede'].unique():
    num_fields = df_potato_2016[df_fields_2016['Afgroede'] == crop_type].shape[0]
    print("There are " + str(num_fields)  + " fields of crop type: " + crop_type)
```

```python

```
