```python
import numpy as np
import xarray as xr
from pathlib import Path
from utils import get_plot_df
from tqdm.autonotebook import tqdm

# Automatically prints execution time for the individual cells
%load_ext autotime

# Automatically reloads functions defined in external files
%load_ext autoreload
%autoreload 2

# Set xarray to use html as display_style
xr.set_options(display_style="html")

# The path to the project (so absoute file paths can be used throughout the notebook)
PROJ_PATH = Path.cwd().parent
```

```python
netcdf_path = (PROJ_PATH / 'data' / 'processed' / 'FieldPolygons2019_stats').with_suffix('.nc')
ds = xr.open_dataset(netcdf_path, engine="h5netcdf")
ds  # Remember to close the dataset before the netcdf file can be rewritten in cells above
```

```python
ds.close()
```

```python
df = ds.to_dataframe()
df = df.reset_index()  # Removes MultiIndex
df = df.drop(columns=['cvr', 'gb', 'gbanmeldt', 'journalnr', 'marknr', 'pass_mode', 'relative_orbit'])
```

```python
pd.set_option('display.max_rows', 100)
df.drop(columns=['polarization']).head(1000000)
```

```python
df[df['date']=='2018-08-01']
```

```python
# Start by finding number of fields, dates, and polarizations
num_fields = len(df['field_id'].unique())
num_dates = len(df['date'].unique())
num_polarizations = len(df['polarization'].unique())

# Get the labels (and ensure that our dataframe is formatted as it is suppposed to be)
for i, date in enumerate(df['date'].unique()):  # Loop over all dates
    # Get lists with afgkode and field_ids for all fields for a single date
    df_date = df[df['date'] == date]   # Extract a df with all values for that date
    y = df_date['afgkode'].iloc[::num_polarizations].values  # Extract 'afgkode' from every N'th row 
    y_field_id = df_date['field_id'].iloc[::num_polarizations].values  # Extract 'field_id' from every N'th row
    
    # Store the lists from the first date
    if i == 0:  
        y_initial = y
        y_field_id_initial = y_field_id
    
    # Check that the lists for every date matches the first date
    assert np.array_equal(y, y_initial)
    assert np.array_equal(y_field_id, y_field_id_initial)
    
# The feature array should have all features (all polarizations for all dates) as a single row per field.
# NOTE: This could be probably be done faster, but this way is simple and easy to understand
print("Converting dataframe to NumPy feature array")
X = np.zeros((num_fields, num_dates*num_polarizations))  # Initialize array
for i, field_id in enumerate(tqdm(y_field_id)):  # Loop over all fields
    df_field = df[df['field_id'] == field_id]  # Extract df for the specific field
    X[i, :] = df_field['stats_mean'].values  # Extract the values ('stats_mean') and insert them into feature array
    # NOTE: Test with using the stats_std also for classification
    #X[i, ??:] = df_field['stats_std'].values

# Print numbers and shapes to give impression of dataset size
print(f"Number of fields: {num_fields}")
print(f"Number of dates: {num_dates}")
print(f"Number of polarizations: {num_polarizations}")
print(f"Shape of feature array: {np.shape(X)}")
print(f"Shape of label array: {np.shape(y)}")
```

```python
X = np.zeros((num_fields, num_dates*num_polarizations))
for i, field_id in enumerate(y_field_id):
    df_field = df[df['field_id'] == field_id]
    X[i, :] = df_field['stats_mean'].values
    
    
#for i, date in enumerate(df['date'].unique()):
#    for j, polarization in enumerate(df['polarization'].unique()):
#        print(polarization)
#        print(j)
#        X[i, j*(i+1)] 
#    break
```

```python
df.iloc[num_fields*num_polarizations]
```

```python
df.iloc[2*num_fields*num_polarizations]
```
