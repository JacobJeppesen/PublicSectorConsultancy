```python pycharm={"is_executing": false}
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;
import xarray as xr
sns.set_style('ticks')
from pathlib import Path
from matplotlib import cm  # For waterfall plot
from matplotlib.ticker import LinearLocator, FormatStrFormatter  # For waterfall plot
from utils import get_plot_df, get_sklearn_df, plot_heatmap_all_polarizations, plot_waterfall_all_polarizations

# Ignore warnings in this notebook
import warnings; warnings.simplefilter('ignore')

# Set matplotlib to plot in notebook
%matplotlib inline

# Automatically prints execution time for the individual cells
%load_ext autotime

# Automatically reloads functions defined in external files
%load_ext autoreload
%autoreload 2

# Set xarray to use html as display_style
xr.set_options(display_style="html")

# The path to the project (so absoute file paths can be used throughout the notebook)
PROJ_PATH = Path.cwd().parent

# Set seed for random generators
RANDOM_SEED = 42

# Seed the random generators
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
```

```python pycharm={"is_executing": false}
netcdf_path = (PROJ_PATH / 'data' / 'processed' / 'FieldPolygons2018_stats').with_suffix('.nc')
ds = xr.open_dataset(netcdf_path, engine="h5netcdf")
ds  # Remember to close the dataset before the netcdf file can be rewritten in cells above
```

```python
ds.close()
```

```python pycharm={"is_executing": false}
df = get_plot_df(polygons_year=2019, 
                 satellite_dates=slice('2019-01-01', '2019-10-31'), 
                 fields='all', 
                 satellite='all', 
                 polarization='VV',
                 crop_type='all')

ALL_CROP_TYPES = df['afgroede'].unique()
print(ALL_CROP_TYPES)
```

```python pycharm={"is_executing": false}
for year in range(2017, 2020):
    print(f"Analyzing {year}")
    df = get_plot_df(polygons_year=year, 
                     satellite_dates=slice(f'{year}-06-10', f'{year}-06-27'), 
                     fields='all', 
                     satellite='all', 
                     polarization='VH')

    #df = df[['satellite', 'stats_mean', 'stats_std', 'stats_min', 'stats_max', 'stats_median']]
    df = df[['satellite', 'stats_mean', 'stats_std']]
    plt.figure(figsize=(6, 6))
    plot = sns.pairplot(df, hue='satellite')
    plot.axes[0,0].set_xlim(-40,0)
    plot.axes[1,1].set_xlim(0,5)
    plt.show()
```

```python
for year in range(2017,2020):
    print(f"Analyzing {year}")
    df = get_plot_df(polygons_year=year, 
                     satellite_dates=slice(f'{year}-06-10', f'{year}-06-25'), 
                     fields='all', 
                     satellite='all', 
                     polarization='VV')

    df = df[['afgroede', 'stats_mean', 'stats_std']]
    df = df[df['stats_mean'] > -20]
    plt.figure(figsize=(6, 6))
    plot = sns.pairplot(df, hue='afgroede')
    plot.axes[0,0].set_xlim(-20,5)
    plot.axes[1,1].set_xlim(0,5)
    plt.show()
```

```python
for year in range(2017,2020):
    print(f"Analyzing {year}")
    df = get_plot_df(polygons_year=year, 
                     satellite_dates=slice(f'{year}-06-10', f'{year}-06-25'), 
                     fields='all', 
                     satellite='S1A', 
                     polarization='VH')

    df = df[['afgroede', 'stats_mean', 'stats_std']]
    df = df[df['stats_mean'] > -30]
    plt.figure(figsize=(6, 6))
    plot = sns.pairplot(df, hue='afgroede')
    plot.axes[0,0].set_xlim(-30,-5)
    plot.axes[1,1].set_xlim(0,5)
    plt.show()
```

```python pycharm={"is_executing": false}
for year in range(2017,2020):
    print(f"Analyzing {year}")
    df = get_plot_df(polygons_year=year, 
                     satellite_dates=slice(f'{year}-01-01', f'{year}-12-31'), 
                     fields='all',#range(100), 
                     satellite='all', 
                     polarization='VH')

    plt.figure(figsize=(24, 8))
    plt.xticks(rotation=90, horizontalalignment='center')
    ax = sns.lineplot(x='date', y='stats_mean', hue='afgroede', data=df, ci='sd')
    plt.show()
```

```python
for year in range(2017,2020):
    print(f"Analyzing {year}")
    df = get_plot_df(polygons_year=year, 
                     satellite_dates=slice(f'{year}-01-01', f'{year}-12-31'), 
                     fields='all',#range(100), 
                     satellite='all', 
                     polarization='VH')

    df = df[df['afgroede'].isin(['Silomajs', 'Vinterraps', 'Skovdrift, alm.'])]
    plt.figure(figsize=(24, 8))
    plt.xticks(rotation=90, horizontalalignment='center')
    ax = sns.lineplot(x='date', y='stats_mean', hue='afgroede', data=df, ci='sd')
    plt.show()
```

```python pycharm={"is_executing": false}
#crop_types = ['Silomajs', 'Vinterraps', 'Skovdrift, alm.']
crop_types = ['Kartofler, stivelses-']
for crop_type in crop_types:
    for year in range(2017, 2020):
        print(f"Plotting {crop_type} from {year}")
        plot_waterfall_all_polarizations(polygons_year=year,
                                         crop_type=crop_type, 
                                         satellite_dates=slice('2016-01-01', '2020-01-01'), 
                                         num_fields=32, 
                                         satellite='all', 
                                         sort_rows=False)
```

```python pycharm={"is_executing": false}
crop_types = ALL_CROP_TYPES #['Silomajs', 'Vinterraps', 'Skovdrift, alm.']
crop_types = ['Kartofler, stivelses-']
for crop_type in crop_types:
    for year in range(2017, 2020):
        print(f"Plotting {crop_type} from {year}")
        plot_heatmap_all_polarizations(polygons_year=year,
                                       crop_type=crop_type,
                                       satellite_dates=slice('2016-01-01', '2020-01-01'),
                                       num_fields=128, 
                                       satellite='all')
```

```python
for year in range(2017, 2020):
    print(f"Plotting {crop_type} from {year}")
    df = get_sklearn_df(polygons_year=year, 
                        satellite_dates=slice(f'{year}-06-21', f'{year}-08-31'), 
                        fields='all',#range(100), 
                        satellite='S1A', 
                        polarization='VH')

    df = df.drop(columns=['field_id', 'afgkode'])
    crop_types = ['Silomajs', 'Vinterraps', 'Skovdrift, alm.']
    df = df[df['afgroede'].isin(crop_types)]
    
    # Get balanced classes
    df_plot_balanced = pd.DataFrame(columns=df.columns)
    for crop_type in crop_types:
        df_plot_balanced = pd.concat([df_plot_balanced, df[df['afgroede'] == crop_type].sample(100)])
        
    # Plot    
    plt.figure(figsize=(2, 2))  # TODO: Why does this not work?
    g = sns.PairGrid(df_plot_balanced, hue='afgroede')
    # Set xlim for each x-axis
    for i in range(len(g.axes[:, 0])):
        g.axes[i, i].set_xlim(-30, -5)
    g.map_diag(sns.kdeplot)
    g.map_lower(plt.scatter)
    g.map_upper(sns.kdeplot)
    g.add_legend()
    plt.show()
```

```python
# NOTE: You have to be very precise with the dates you choose - they must be close to eachother, and the number of dates per year must match
crop_type = 'Vinterraps'
df_year = []
for year in range(2017, 2020):
    print(f"Getting df of {crop_type} from {year}")
    df = get_sklearn_df(polygons_year=year, 
                        satellite_dates=slice(f'{year}-05-08', f'{year}-09-18'), 
                        fields='all',#range(100), 
                        satellite='S1A', 
                        polarization='VH')

    df = df[df['afgroede'] == crop_type]
    df = df.drop(columns=['field_id', 'afgkode', 'afgroede'])
    df = df.sample(100)
    df.insert(0, 'year',year)
    df['year'] = ['$%s$' % x for x in df['year']]  # https://github.com/mwaskom/seaborn/issues/1653
    df_year.append(df)

for i, df in enumerate(df_year):
    print(f"The first date is {df.columns.values[1]} and last date is {df.columns.values[-1]} for year {i} (total of {len(df.columns[1:])} dates)")
    new_column_names = ['year'] + [f'date_{i}' for i in range(len(df.columns[1:]))]
    df.columns = new_column_names
    df_year[i] = df
```

```python
# Plot    
df = pd.concat(df_year)
g = sns.PairGrid(df, hue='year')
# Set xlim for each x-axis
for i in range(len(g.axes[:, 0])):
    g.axes[i, i].set_xlim(-30, -5)
g.map_diag(sns.kdeplot)
g.map_lower(plt.scatter)
g.map_upper(sns.kdeplot)
g.add_legend()
plt.show()
```

```python
# NOTE: You have to be very precise with the dates you choose - they must be close to eachother, and the number of dates per year must match
crop_type = 'Silomajs'
df_year = []
for year in range(2017, 2020):
    print(f"Getting df of {crop_type} from {year}")
    df = get_sklearn_df(polygons_year=year, 
                        satellite_dates=slice(f'{year}-05-08', f'{year}-09-18'), 
                        fields='all',#range(100), 
                        satellite='S1A', 
                        polarization='VH')

    df = df[df['afgroede'] == crop_type]
    df = df.drop(columns=['field_id', 'afgkode', 'afgroede'])
    df = df.sample(100)
    df.insert(0, 'year',year)
    df['year'] = ['$%s$' % x for x in df['year']]  # https://github.com/mwaskom/seaborn/issues/1653
    df_year.append(df)

for i, df in enumerate(df_year):
    print(f"The first date is {df.columns.values[1]} and last date is {df.columns.values[-1]} for year {i} (total of {len(df.columns[1:])} dates)")
    new_column_names = ['year'] + [f'date_{i}' for i in range(len(df.columns[1:]))]
    df.columns = new_column_names
    df_year[i] = df
```

```python
# Plot    
df = pd.concat(df_year)
g = sns.PairGrid(df, hue='year')
# Set xlim for each x-axis
for i in range(len(g.axes[:, 0])):
    g.axes[i, i].set_xlim(-30, -5)
g.map_diag(sns.kdeplot)
g.map_lower(plt.scatter)
g.map_upper(sns.kdeplot)
g.add_legend()
plt.show()
```

```python
# NOTE: You have to be very precise with the dates you choose - they must be close to eachother, and the number of dates per year must match
crop_type = 'Silomajs'
df_year = []
for year in range(2017, 2020):
    print(f"Getting df of {crop_type} from {year}")
    df = get_sklearn_df(polygons_year=year, 
                        satellite_dates=slice(f'{year}-05-08', f'{year}-07-19'), 
                        fields='all',#range(100), 
                        satellite='all', 
                        polarization='VH')

    df = df[df['afgroede'] == crop_type]
    df = df.drop(columns=['field_id', 'afgkode', 'afgroede'])
    df = df.sample(100)
    df.insert(0, 'year',year)
    df['year'] = ['$%s$' % x for x in df['year']]  # https://github.com/mwaskom/seaborn/issues/1653
    df_year.append(df)

for i, df in enumerate(df_year):
    print(f"The first date is {df.columns.values[1]} and last date is {df.columns.values[-1]} for year {i} (total of {len(df.columns[1:])} dates)")
    new_column_names = ['year'] + [f'date_{i}' for i in range(len(df.columns[1:]))]
    df.columns = new_column_names
    df_year[i] = df
```

```python
# Plot    
df = pd.concat(df_year)
g = sns.PairGrid(df, hue='year')
# Set xlim for each x-axis
for i in range(len(g.axes[:, 0])):
    g.axes[i, i].set_xlim(-30, -5)
g.map_diag(sns.kdeplot)
g.map_lower(plt.scatter)
g.map_upper(sns.kdeplot)
g.add_legend()
plt.show()
```

```python pycharm={"is_executing": false}
# Idea for violin plot: Use it to compare individual dates on the x-axis. 
# For instance have 5 dates and 2 crop types, and then use x=dates, y=stats-mean, hue=afgroede. 
# This would give you a comparison of the distributions for two crop types for five different dates. 
# That might be useful.
```

```python
df = get_plot_df(polygons_year=2019, 
                 satellite_dates=slice('2016-01-01', '2020-01-01'), 
                 fields='all',#range(100), 
                 satellite='all', 
                 polarization='VH',
                 netcdf_path=netcdf_path)

df = df[df['afgroede'].isin(['Skovdrift, alm.', 'Silomajs', 'Vinterraps'])]
df = df.rename(columns={'afgroede': 'Crop type'})
df.loc[df['Crop type'] == 'Skovdrift, alm.', 'Crop type'] = 'Forestry'
df.loc[df['Crop type'] == 'Silomajs', 'Crop type'] = 'Maize'
df.loc[df['Crop type'] == 'Vinterraps', 'Crop type'] = 'Rapeseed'

plt.figure(figsize=(13.5, 4.5))
plt.xticks(rotation=90, horizontalalignment='center')
ax = sns.lineplot(x='date', y='stats_mean', hue='Crop type', hue_order=['Forestry', 'Maize', 'Rapeseed'], 
                  data=df.sort_index(ascending=False), ci='sd')
ax.set_ylabel('Mean VH backscattering [dB]')
ax.set_xlabel('')
#ax.set_ylim(-29, -9)
ax.margins(x=0.01)

# Only show every n'th tick on the x-axis
ticks_divider = 1
dates = df['date'].unique()
num_dates = len(dates)
xticks = range(0, num_dates)[::ticks_divider] 
xticklabels = dates[::ticks_divider]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, horizontalalignment='center')

# Save the figure
save_path = PROJ_PATH / 'reports' / 'figures' / 'TemporalVariationOverview.pdf'
plt.tight_layout()
plt.savefig(save_path)
```

```python
df = get_plot_df(polygons_year=2018, 
                 satellite_dates=slice('2016-01-01', '2020-01-01'), 
                 fields='all',#range(100), 
                 satellite='all', 
                 polarization='VH',
                 netcdf_path=netcdf_path2018)

df = df[df['afgroede'].isin(['Skovdrift, alm.', 'Silomajs', 'Vinterraps'])]
df = df.rename(columns={'afgroede': 'Crop type'})
df.loc[df['Crop type'] == 'Skovdrift, alm.', 'Crop type'] = 'Forestry'
df.loc[df['Crop type'] == 'Silomajs', 'Crop type'] = 'Maize'
df.loc[df['Crop type'] == 'Vinterraps', 'Crop type'] = 'Rapeseed'

plt.figure(figsize=(13.5, 4.5))
plt.xticks(rotation=90, horizontalalignment='center')
ax = sns.lineplot(x='date', y='stats_mean', hue='Crop type', hue_order=['Forestry', 'Maize', 'Rapeseed'], 
                  data=df.sort_index(ascending=False), ci='sd')
ax.set_ylabel('Mean VH backscattering [dB]')
ax.set_xlabel('')
#ax.set_ylim(-29, -9)
ax.margins(x=0.01)

# Only show every n'th tick on the x-axis
ticks_divider = 1
dates = df['date'].unique()
num_dates = len(dates)
xticks = range(0, num_dates)[::ticks_divider] 
xticklabels = dates[::ticks_divider]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, horizontalalignment='center')

# Save the figure
save_path = PROJ_PATH / 'reports' / 'figures' / 'TemporalVariationOverview.pdf'
plt.tight_layout()
plt.savefig(save_path)
```

```python
df = get_plot_df(polygons_year=2017, 
                 satellite_dates=slice('2016-01-01', '2020-01-01'), 
                 fields='all',#range(100), 
                 satellite='all', 
                 polarization='VH',
                 netcdf_path=netcdf_path2017)

df = df[df['afgroede'].isin(['Skovdrift, alm.', 'Silomajs', 'Vinterraps'])]
df = df.rename(columns={'afgroede': 'Crop type'})
df.loc[df['Crop type'] == 'Skovdrift, alm.', 'Crop type'] = 'Forestry'
df.loc[df['Crop type'] == 'Silomajs', 'Crop type'] = 'Maize'
df.loc[df['Crop type'] == 'Vinterraps', 'Crop type'] = 'Rapeseed'

plt.figure(figsize=(13.5, 4.5))
plt.xticks(rotation=90, horizontalalignment='center')
ax = sns.lineplot(x='date', y='stats_mean', hue='Crop type', hue_order=['Forestry', 'Maize', 'Rapeseed'], 
                  data=df.sort_index(ascending=False), ci='sd')
ax.set_ylabel('Mean VH backscattering [dB]')
ax.set_xlabel('')
#ax.set_ylim(-29, -9)
ax.margins(x=0.01)

# Only show every n'th tick on the x-axis
ticks_divider = 1
dates = df['date'].unique()
num_dates = len(dates)
xticks = range(0, num_dates)[::ticks_divider] 
xticklabels = dates[::ticks_divider]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, horizontalalignment='center')

# Save the figure
save_path = PROJ_PATH / 'reports' / 'figures' / 'TemporalVariationOverview.pdf'
plt.tight_layout()
plt.savefig(save_path)
```

```python
df = get_plot_df(polygons_year=2019, 
                 satellite_dates=slice('2018-07-01', '2019-11-01'), 
                 fields='all',#range(100), 
                 satellite='S1A', 
                 polarization='VH',
                 netcdf_path=netcdf_path)

df = df[df['afgroede'].isin(['Skovdrift, alm.', 'Silomajs', 'Vinterraps'])]
df = df.rename(columns={'afgroede': 'Crop type'})
df.loc[df['Crop type'] == 'Skovdrift, alm.', 'Crop type'] = 'Forestry'
df.loc[df['Crop type'] == 'Silomajs', 'Crop type'] = 'Maize'
df.loc[df['Crop type'] == 'Vinterraps', 'Crop type'] = 'Rapeseed'

plt.figure(figsize=(13.5, 4.5))
plt.xticks(rotation=90, horizontalalignment='center')
ax = sns.lineplot(x='date', y='stats_mean', hue='Crop type', hue_order=['Forestry', 'Maize', 'Rapeseed'], 
                  data=df.sort_index(ascending=False), ci='sd')
ax.set_ylabel('Mean VH backscattering [dB]')
ax.set_xlabel('')
#ax.set_ylim(-29, -9)
ax.margins(x=0.01)

# Only show every n'th tick on the x-axis
ticks_divider = 1
dates = df['date'].unique()
num_dates = len(dates)
xticks = range(0, num_dates)[::ticks_divider] 
xticklabels = dates[::ticks_divider]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, horizontalalignment='center')

# Save the figure
save_path = PROJ_PATH / 'reports' / 'figures' / 'TemporalVariationOverview_S1A.pdf'
plt.tight_layout()
plt.savefig(save_path)
```

```python
def plot_and_save_waterfall(crop_type, crop_name, save_path, fontsize=12):
    df = get_plot_df(polygons_year=2019, 
                     satellite_dates=slice('2018-07-01', '2019-11-01'), 
                     fields='all',#range(100), 
                     satellite='all', 
                     crop_type=crop_type,
                     polarization='VH')

    df = df.dropna()

    # Get the dates (needed later for plotting)
    num_fields = 32
    dates = df['date'].unique()
    num_dates = len(dates)
    sort_rows = False

    # Pivot the df (https://stackoverflow.com/a/37790707/12045808)
    df = df.pivot(index='field_id', columns='date', values='stats_mean')

    # Drop fields having any date with a nan value, and pick num_fields from the remainder
    df = df.dropna().sample(n=num_fields, random_state=1)

    if sort_rows:
        # Sort by sum of each row
        df = df.reset_index()
        df = df.drop(columns=['field_id'])
        idx = df.sum(axis=1).sort_values(ascending=False).index
        df = df.iloc[idx]

    # Get the min and max values depending on polarization
    vmin_cm, vmax_cm = -25, -10
    vmin_z, vmax_z = -30, -5

    # Make data.
    x = np.linspace(1, num_dates, num_dates)  # Dates
    y = np.linspace(1, num_fields, num_fields)  # Fields
    X,Y = np.meshgrid(x, y)
    Z = df.to_numpy()

    # Plot the surface.
    fig = plt.figure(figsize=(7, 7))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, vmin=vmin_cm, vmax=vmax_cm,
                           linewidth=0, antialiased=False)

    # Set title 
    ax.set_title(f"Temporal evolution of {crop_name}", fontsize=fontsize+2)

    # Set angle (https://stackoverflow.com/a/47610615/12045808)
    ax.view_init(25, 280)

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # Customize the z axis (backscattering value)
    ax.zaxis.set_major_locator(LinearLocator(6))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_zlim(vmin_z, vmax_z)
    for tick in ax.zaxis.get_major_ticks():
        tick.label1.set_horizontalalignment('left')
        tick.label1.set_fontsize(fontsize) 

    # Customize the x axis (dates)
    # If >10 dates, skip every second tick, if >20 dates, skip every third ... 
    ticks_divider = int(np.ceil(num_dates/10))  
    xticks = range(1, num_dates+1)[::ticks_divider]  # Array must be starting at 1
    xticklabels = dates[::ticks_divider]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=75, horizontalalignment='right')

    # Customize the y axis (field ids)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_horizontalalignment('left')
        tick.label1.set_verticalalignment('bottom')
        tick.label1.set_rotation(-5)

    # Set viewing distance (important to not cut off labels)
    ax.dist = 11

    # Set labels (the spaces are a hacky way to get the labels to be at the correct position)
    ax.set_ylabel('              Field', labelpad=18, fontsize=fontsize)
    ax.set_zlabel('Mean VH backscattering [dB]         ', labelpad=44, fontsize=fontsize)
    
    # Set tick size
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    #plt.zticks(fontsize=fontsize)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
```

```python
save_path = PROJ_PATH / 'reports' / 'figures' / 'WaterfallForestry.pdf'
plot_and_save_waterfall('Skovdrift, alm.', 'forestry', save_path, fontsize=11)
```

```python
save_path = PROJ_PATH / 'reports' / 'figures' / 'WaterfallMaize.pdf'
plot_and_save_waterfall('Silomajs', 'maize', save_path, fontsize=11)
```

```python
save_path = PROJ_PATH / 'reports' / 'figures' / 'WaterfallRapeseed.pdf'
plot_and_save_waterfall('Vinterraps', 'rapeseed', save_path, fontsize=11)
```

```python
save_path = PROJ_PATH / 'reports' / 'figures' / 'WaterfallPotatoStivelses.pdf'
plot_and_save_waterfall('Kartofler, stivelses-', 'Kartofler, stivelses-', save_path, fontsize=11)
```

```python
save_path = PROJ_PATH / 'reports' / 'figures' / 'WaterfallPotatoSpise.pdf'
plot_and_save_waterfall('Kartofler, spise-', 'Kartofler, spise-', save_path, fontsize=11)
```

```python
def plot_and_save_heatmap(crop_type, crop_name, save_path, fontsize=12):
    fig = plt.figure(figsize=(4, 5))
    
    df = get_plot_df(polygons_year=2019, 
                     satellite_dates=slice('2018-07-01', '2019-11-01'), 
                     fields='all',#range(100), 
                     satellite='all', 
                     crop_type=crop_type,
                     polarization='VH')

    # Get the dates (needed later for plotting)
    num_fields = 128
    dates = df['date'].unique()
    num_dates = len(dates)
    sort_rows = False
    
    # Pivot the df (https://stackoverflow.com/a/37790707/12045808)
    df = df.pivot(index='field_id', columns='date', values='stats_mean')

    # Drop fields having any date with a nan value, and pick num_fields from the remainder
    df = df.dropna()
    if num_fields > df.shape[0]:
        num_fields = df.shape[0]
        print(f"Only {num_fields} fields were available for plotting")
    df = df.sample(n=num_fields, random_state=1)

    if sort_rows:
        # Sort by sum of each row
        df = df.reset_index()
        df = df.drop(columns=['Field'])
        idx = df.sum(axis=1).sort_values(ascending=False).index
        df = df.iloc[idx]

    # Get the min and max values depending on polarization
    vmin, vmax = -25, -10 

    ax = sns.heatmap(df, linewidths=0, linecolor=None, vmin=vmin, vmax=vmax, yticklabels=False, 
                     cmap=cm.coolwarm)
    
    # Pad label for cbar (https://stackoverflow.com/questions/52205416/moving-label-of-seaborn-colour-bar)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Mean VH backscattering [dB]', labelpad=10, fontsize=fontsize)
    
    # Customize the x axis (dates)
    ticks_divider = 15
    xticks = range(1, num_dates+1)[::ticks_divider]
    xticklabels = dates[::ticks_divider]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=90, horizontalalignment='center', fontsize=fontsize)
    
    # Fix labels
    ax.set_xlabel('')
    ax.set_ylabel('Field', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    fig = ax.get_figure()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
```

```python
save_path = PROJ_PATH / 'reports' / 'figures' / 'HeatmapForestry.pdf'
plot_and_save_heatmap('Skovdrift, alm.', 'forestry', save_path, fontsize=10)
```

```python
save_path = PROJ_PATH / 'reports' / 'figures' / 'HeatmapMaize.pdf'
plot_and_save_heatmap('Silomajs', 'maize', save_path, fontsize=10)
```

```python
save_path = PROJ_PATH / 'reports' / 'figures' / 'HeatmapRapeseed.pdf'
plot_and_save_heatmap('Vinterraps', 'rapeseed', save_path, fontsize=10)
```

```python
save_path = PROJ_PATH / 'reports' / 'figures' / 'HeatmapPotatoStivelses.pdf'
plot_and_save_heatmap('Kartofler, stivelses-', 'Kartofler, stivelses-', save_path, fontsize=10)
```

```python
save_path = PROJ_PATH / 'reports' / 'figures' / 'HeatmapPotatoSpise.pdf'
plot_and_save_heatmap('Kartofler, spise-', 'Kartofler, spise-', save_path, fontsize=10)
```
