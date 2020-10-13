```python
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
from utils import get_df, get_plot_df, get_sklearn_df, plot_heatmap_all_polarizations, plot_waterfall_all_polarizations

# Ignore warnings in this notebook
import warnings; warnings.simplefilter('ignore')

# Set matplotlib to plot in notebook
%matplotlib inline

# Automatically prints execution time for the individual cells
%load_ext autotime

# Automatically reloads functions defined in external files
%load_ext autoreload
%autoreload 2

# The path to the project (so absoute file paths can be used throughout the notebook)
PROJ_PATH = Path.cwd().parent

# Set seed for random generators
RANDOM_SEED = 42

# Seed the random generators
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
```

The Excel sheet had attributes cleaned (ie. one empty row was removed and a typo in one of the years were corrected). Then the field_ids were found in QGIS based on the Markbloknr. and MARK NR., and added as an additional column in the Excel sheet.

```python
# Load Excel sheet with infected fields
infected_fields_excel_sheet_path = PROJ_PATH / 'data' / 'external' / 'Projekt_KCN_med_AU_renset_for_navn_til_pandas.xlsx'
df_infected_fields = pd.read_excel(infected_fields_excel_sheet_path)

# Show the resulting dataframe
df_infected_fields
```

```python
for year in range(2017,2020):
    # Load dataframe
    df = get_df(polygons_year=year, 
                satellite_dates=slice(f'{year}-01-01', f'{year}-12-31'), 
                fields='all',#range(100), 
                satellite='all', 
                polarization='VH')

    # Get field ids of the infected fields for the specific year
    df_infected_fields_year = df_infected_fields[df_infected_fields['Sæson'] == year] 
    infected_field_ids = df_infected_fields_year['field_id'].values
    
    # Add 'infected' column with True/False if Infected/Not-infected
    df['Infected'] = ['True' if x in infected_field_ids else 'Not controlled' for x in df['field_id']]
    
    # Loop over the crop types and plot the temporal evolution
    infected_crop_types = df[df['Infected'] == 'True']['afgroede'].unique()
    for i, crop_type in enumerate(infected_crop_types):
        print(f"Analyzing {year} season for crop type: {crop_type}")
        df_plot = df[df['afgroede'] == infected_crop_types[i]]
        num_total_fields = len(df_plot['field_id'].unique())
        num_infected = len(df_plot[df_plot['Infected'] == 'True']['field_id'].unique())

        # Format the dataframe to work well with Seaborn for plotting
        df_plot['date'] = df_plot['date'].dt.strftime('%Y-%m-%d')
        df_plot['field_id'] = ['$%s$' % x for x in df_plot['field_id']]  # https://github.com/mwaskom/seaborn/issues/1653
        df_plot['afgkode'] = ['$%s$' % x for x in df_plot['afgkode']]  # https://github.com/mwaskom/seaborn/issues/1653
        df_plot['relative_orbit'] = ['$%s$' % x for x in df_plot['relative_orbit']]  # https://github.com/mwaskom/seaborn/issues/1653

        # Plot
        plt.figure(figsize=(24, 8))
        plt.xticks(rotation=90, horizontalalignment='center')
        plt.title(f"{year}, {crop_type}, {num_infected} confirmed infected of a total of {num_total_fields} fields")
        ax = sns.lineplot(x='date', y='stats_mean', hue='Infected', data=df_plot, ci='sd')
        plt.show()
```

```python

```
