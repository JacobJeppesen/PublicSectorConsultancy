```python pycharm={"is_executing": false}
import matplotlib.pyplot as plt
import seaborn as sns;
import xarray as xr
sns.set_style('ticks')
from pathlib import Path
from utils import get_plot_df, plot_heatmap_all_polarizations, plot_waterfall_all_polarizations

# Ignore warnings in this notebook
import warnings; warnings.simplefilter('ignore')

# Automatically prints execution time for the individual cells
%load_ext autotime

# Automatically reloads functions defined in external files
%load_ext autoreload
%autoreload 2

# Set xarray to use html as display_style
xr.set_options(display_style="html")

# The path to the project (so absoute file paths can be used throughout the notebook)
PROJ_PATH = Path.cwd().parent

# Define which field polygons should be used for analysis (2017 to 2019 seem to follow the same metadata format)
FIELD_POLYGONS = ['FieldPolygons2017', 'FieldPolygons2018', 'FieldPolygons2019']
```

```python pycharm={"is_executing": false}
netcdf_path = (PROJ_PATH / 'data' / 'processed' / 'FieldPolygons2019_stats').with_suffix('.nc')
ds = xr.open_dataset(netcdf_path, engine="h5netcdf")
ds  # Remember to close the dataset before the netcdf file can be rewritten in cells above
```

```python pycharm={"is_executing": false}
ds.close()
```

```python pycharm={"is_executing": false}
df = get_plot_df(polygons_year=2019, 
                 satellite_dates=slice('2019-01-01', '2019-01-31'), 
                 fields='all', 
                 satellite='all', 
                 polarization='VV',
                 crop_type='all',
                 netcdf_path=netcdf_path)

ALL_CROP_TYPES = df['afgroede'].unique()
print(ALL_CROP_TYPES)
```

```python pycharm={"is_executing": false}
df = get_plot_df(polygons_year=2019, 
                 satellite_dates=slice('2019-01-01', '2019-03-31'), 
                 fields='all', 
                 satellite='all', 
                 polarization='VV',
                 netcdf_path=netcdf_path)

print("Types of pass-mode: {}".format(df['pass_mode'].unique()))

plt.figure(figsize=(24, 8))
ax = sns.scatterplot(x='stats_mean', y='stats_std', hue='satellite', data=df)
```

```python pycharm={"is_executing": false}
df = get_plot_df(polygons_year=2019, 
                 satellite_dates=slice('2019-01-01', '2019-03-31'), 
                 fields='all', 
                 satellite='all', 
                 polarization='VV',
                 netcdf_path=netcdf_path)

df = df[['satellite', 'stats_mean', 'stats_std', 'stats_min', 'stats_max', 'stats_median']]
plt.figure(figsize=(24, 24))
ax = sns.pairplot(df, hue='satellite')
```

```python pycharm={"is_executing": false}
df = get_plot_df(polygons_year=2019, 
                 satellite_dates=slice('2019-06-20', '2019-06-22'), 
                 fields='all', 
                 satellite='all', 
                 polarization='VH',
                 netcdf_path=netcdf_path)

df = df[['afgroede', 'stats_mean', 'stats_std', 'stats_min', 'stats_max', 'stats_median']]
plt.figure(figsize=(24, 24))
ax = sns.pairplot(df, hue='afgroede')
```

```python pycharm={"is_executing": false}
df = get_plot_df(polygons_year=2019, 
                 satellite_dates=slice('2018-01-01', '2019-12-31'), 
                 fields='all',#range(100), 
                 satellite='all', 
                 polarization='VV',
                 netcdf_path=netcdf_path)

plt.figure(figsize=(24, 8))
plt.xticks(rotation=90, horizontalalignment='center')
ax = sns.lineplot(x='date', y='stats_mean', hue='afgroede', data=df, ci='sd')
```

```python pycharm={"is_executing": false}
df = get_plot_df(polygons_year=2019, 
                 satellite_dates=slice('2018-01-01', '2019-12-31'), 
                 fields='all',#range(100), 
                 satellite='all', 
                 polarization='VH',
                 netcdf_path=netcdf_path)

plt.figure(figsize=(24, 8))
plt.xticks(rotation=90, horizontalalignment='center')
ax = sns.lineplot(x='date', y='stats_mean', hue='afgroede', data=df, ci='sd')
```

```python pycharm={"is_executing": false}
df = get_plot_df(polygons_year=2019, 
                 satellite_dates=slice('2018-01-01', '2019-12-31'), 
                 fields='all',#range(100), 
                 satellite='S1A', 
                 polarization='VH',
                 netcdf_path=netcdf_path)

df = df[df['afgroede'].isin(['Silomajs', 'Vinterhvede', 'Kartofler, stivelses-'])]
plt.figure(figsize=(24, 8))
plt.xticks(rotation=90, horizontalalignment='center')
ax = sns.lineplot(x='date', y='stats_mean', hue='afgroede', data=df, ci='sd')
```

```python pycharm={"is_executing": false}
for crop_type in ALL_CROP_TYPES:
    print(f"Plotting {crop_type}")
    plot_waterfall_all_polarizations(crop_type=crop_type, 
                                     satellite_dates=slice('2018-01-01', '2019-12-31'), 
                                     num_fields=32, 
                                     satellite='S1A', 
                                     sort_rows=False, 
                                     netcdf_path=netcdf_path)
```

```python pycharm={"is_executing": false}
for crop_type in ALL_CROP_TYPES:  
    print(f"Plotting {crop_type}")
    plot_heatmap_all_polarizations(crop_type=crop_type,
                                   satellite_dates=slice('2018-01-01', '2019-12-31'),
                                   num_fields=128, 
                                   satellite='all', 
                                   netcdf_path=netcdf_path)
```

```python pycharm={"is_executing": false}
# Idea for violin plot: Use it to compare individual dates on the x-axis. 
# For instance have 5 dates and 2 crop types, and then use x=dates, y=stats-mean, hue=afgroede. 
# This would give you a comparison of the distributions for two crop types for five different dates. 
# That might be useful.
```
