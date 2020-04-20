```python
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns

from pathlib import Path
from tqdm.autonotebook import tqdm
from sklearn.model_selection import train_test_split         # Split data into train and test set

from utils import evaluate_classifier, get_sklearn_df 

# Automatically prints execution time for the individual cells
%load_ext autotime

# Automatically reloads functions defined in external files
%load_ext autoreload
%autoreload 2

# Set xarray to use html as display_style
xr.set_options(display_style="html")

# Tell matplotlib to plot directly in the notebook
%matplotlib inline  

# The path to the project (so absoute file paths can be used throughout the notebook)
PROJ_PATH = Path.cwd().parent

# Set seed for random generators
RANDOM_SEED = 42
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
# Convert the xarray dataset to pandas dataframe
df = ds.to_dataframe()
df = df.reset_index()  # Removes MultiIndex
df = df.drop(columns=['cvr', 'gb', 'gbanmeldt', 'journalnr', 'marknr', 'pass_mode', 'relative_orbit'])
df = df.dropna()
```

```python
year = 2018
for i in range(7, 24, 1):
    month = (i % 12) + 1
    if month == 1:
        year += 1
        
    print(f"--------------------------------------------------------------------------------------------------")
    print(f"Dataset from 2018-07-01 to {year}-{month:02}-01")
    df_sklearn = get_sklearn_df(polygons_year=2019, 
                                satellite_dates=slice('2018-07-01', f'{year}-{month:02}-01'), 
                                fields='all', 
                                satellite='all', 
                                polarization='all',
                                crop_type='all',
                                netcdf_path=netcdf_path)

    df_sklearn = df_sklearn[df_sklearn['afgroede'].isin(['Vårbyg', 'Vinterhvede', 'Silomajs', 'Vinterraps', 
                                                         'Vinterbyg', 'Vårhavre', 'Vinterhybridrug'])]
    crop_codes = df_sklearn['afgkode'].unique()
    mapping_dict = {}
    class_names = [] 

    for i, crop_code in enumerate(crop_codes):
        mapping_dict[crop_code] = i
        crop_type = df_sklearn[df_sklearn['afgkode'] == crop_code].head(1)['afgroede'].values[0]
        class_names.append(crop_type)

    df_sklearn_remapped = df_sklearn.copy()
    df_sklearn_remapped['afgkode'] = df_sklearn_remapped['afgkode'].map(mapping_dict)
    #print(f"Crop types: {class_names}")

    array = df_sklearn_remapped.values

    # Define the independent variables as features.
    X = np.float32(array[:,3:])  # The features 

    # Define the target (dependent) variable as labels.
    y = np.int8(array[:,1])  # The column 'afgkode'

    # Create a train/test split using 30% test size.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

    #print(f"Train samples:      {len(y_train)}")
    #print(f"Test samples:       {len(y_test)}")
    #print(f"Number of features: {len(X[0,:])}")

    from sklearn.linear_model import LogisticRegression          

    # Instantiate and evaluate classifier
    clf = LogisticRegression(solver='lbfgs', multi_class='auto', n_jobs=10, max_iter=1000)
    clf_trained = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, feature_scale=True, plot_confusion_matrix=False)
```

```python
year = 2018
for i in range(7, 24, 1):
    month = (i % 12) + 1
    if month == 1:
        year += 1
        
    print(f"--------------------------------------------------------------------------------------------------")
    print(f"Dataset from 2018-07-01 to {year}-{month:02}-01")
    df_sklearn = get_sklearn_df(polygons_year=2019, 
                                satellite_dates=slice('2018-07-01', f'{year}-{month:02}-01'), 
                                fields='all', 
                                satellite='all', 
                                polarization='all',
                                crop_type='all',
                                netcdf_path=netcdf_path)

    df_sklearn = df_sklearn[df_sklearn['afgroede'].isin(['Vårbyg', 'Vinterhvede', 'Silomajs', 'Vinterraps', 
                                                         'Vinterbyg', 'Vårhavre', 'Vinterhybridrug'])]
    crop_codes = df_sklearn['afgkode'].unique()
    mapping_dict = {}
    class_names = [] 

    for i, crop_code in enumerate(crop_codes):
        mapping_dict[crop_code] = i
        crop_type = df_sklearn[df_sklearn['afgkode'] == crop_code].head(1)['afgroede'].values[0]
        class_names.append(crop_type)

    df_sklearn_remapped = df_sklearn.copy()
    df_sklearn_remapped['afgkode'] = df_sklearn_remapped['afgkode'].map(mapping_dict)
    #print(f"Crop types: {class_names}")

    array = df_sklearn_remapped.values

    # Define the independent variables as features.
    X = np.float32(array[:,3:])  # The features 

    # Define the target (dependent) variable as labels.
    y = np.int8(array[:,1])  # The column 'afgkode'

    # Create a train/test split using 30% test size.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #print(f"Train samples:      {len(y_train)}")
    #print(f"Test samples:       {len(y_test)}")
    #print(f"Number of features: {len(X[0,:])}")

    from sklearn.linear_model import LogisticRegressionCV          

    # Instantiate and evaluate classifier
    clf = LogisticRegressionCV(solver='lbfgs', multi_class='auto', cv=10, n_jobs=10, random_state=RANDOM_SEED, max_iter=1000)
    clf_trained = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, feature_scale=True, plot_confusion_matrix=False)
```

```python
year = 2018
for i in range(7, 24, 1):
    month = (i % 12) + 1
    if month == 1:
        year += 1
        
    print(f"--------------------------------------------------------------------------------------------------")
    print(f"Dataset from 2018-07-01 to {year}-{month:02}-01")
    df_sklearn = get_sklearn_df(polygons_year=2019, 
                                satellite_dates=slice('2018-07-01', f'{year}-{month:02}-01'), 
                                fields='all', 
                                satellite='all', 
                                polarization='all',
                                crop_type='all',
                                netcdf_path=netcdf_path)

    df_sklearn = df_sklearn[df_sklearn['afgroede'].isin(['Vårbyg', 'Vinterhvede', 'Silomajs', 'Vinterraps', 
                                                         'Vinterbyg', 'Vårhavre', 'Vinterhybridrug'])]
    crop_codes = df_sklearn['afgkode'].unique()
    mapping_dict = {}
    class_names = [] 

    for i, crop_code in enumerate(crop_codes):
        mapping_dict[crop_code] = i
        crop_type = df_sklearn[df_sklearn['afgkode'] == crop_code].head(1)['afgroede'].values[0]
        class_names.append(crop_type)

    df_sklearn_remapped = df_sklearn.copy()
    df_sklearn_remapped['afgkode'] = df_sklearn_remapped['afgkode'].map(mapping_dict)
    #print(f"Crop types: {class_names}")

    array = df_sklearn_remapped.values

    # Define the independent variables as features.
    X = np.float32(array[:,3:])  # The features 

    # Define the target (dependent) variable as labels.
    y = np.int8(array[:,1])  # The column 'afgkode'

    # Create a train/test split using 30% test size.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #print(f"Train samples:      {len(y_train)}")
    #print(f"Test samples:       {len(y_test)}")
    #print(f"Number of features: {len(X[0,:])}")

    from sklearn.linear_model import LogisticRegressionCV          

    # Instantiate and evaluate classifier
    clf = LogisticRegressionCV(solver='lbfgs', multi_class='auto', n_jobs=10, cv=10, random_state=RANDOM_SEED, max_iter=1000, class_weight='balanced')
    clf_trained = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, feature_scale=True, plot_confusion_matrix=False)
```

```python
year = 2018
for i in range(7, 24, 1):
    month = (i % 12) + 1
    if month == 1:
        year += 1
        
    print(f"--------------------------------------------------------------------------------------------------")
    print(f"Dataset from 2018-07-01 to {year}-{month:02}-01")
    df_sklearn = get_sklearn_df(polygons_year=2019, 
                                satellite_dates=slice('2018-07-01', f'{year}-{month:02}-01'), 
                                fields='all', 
                                satellite='all', 
                                polarization='all',
                                crop_type='all',
                                netcdf_path=netcdf_path)

    df_sklearn = df_sklearn[df_sklearn['afgroede'].isin(['Vårbyg', 'Vinterhvede', 'Silomajs', 'Vinterraps', 
                                                         'Vinterbyg', 'Vårhavre', 'Vinterhybridrug'])]
    crop_codes = df_sklearn['afgkode'].unique()
    mapping_dict = {}
    class_names = [] 

    for i, crop_code in enumerate(crop_codes):
        mapping_dict[crop_code] = i
        crop_type = df_sklearn[df_sklearn['afgkode'] == crop_code].head(1)['afgroede'].values[0]
        class_names.append(crop_type)

    df_sklearn_remapped = df_sklearn.copy()
    df_sklearn_remapped['afgkode'] = df_sklearn_remapped['afgkode'].map(mapping_dict)
    #print(f"Crop types: {class_names}")

    array = df_sklearn_remapped.values

    # Define the independent variables as features.
    X = np.float32(array[:,3:])  # The features 

    # Define the target (dependent) variable as labels.
    y = np.int8(array[:,1])  # The column 'afgkode'

    # Create a train/test split using 30% test size.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

    #print(f"Train samples:      {len(y_train)}")
    #print(f"Test samples:       {len(y_test)}")
    #print(f"Number of features: {len(X[0,:])}")

    from sklearn.svm import SVC   

    # Instantiate and evaluate classifier
    clf = SVC(kernel='rbf')
    clf_trained = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, feature_scale=True, plot_confusion_matrix=False)
```

```python
year = 2018
for i in range(7, 24, 1):
    month = (i % 12) + 1
    if month == 1:
        year += 1
        
    print(f"--------------------------------------------------------------------------------------------------")
    print(f"Dataset from 2018-07-01 to {year}-{month:02}-01")
    df_sklearn = get_sklearn_df(polygons_year=2019, 
                                satellite_dates=slice('2018-07-01', f'{year}-{month:02}-01'), 
                                fields='all', 
                                satellite='all', 
                                polarization='all',
                                crop_type='all',
                                netcdf_path=netcdf_path)

    df_sklearn = df_sklearn[df_sklearn['afgroede'].isin(['Vårbyg', 'Vinterhvede', 'Silomajs', 'Vinterraps', 
                                                         'Vinterbyg', 'Vårhavre', 'Vinterhybridrug'])]
    crop_codes = df_sklearn['afgkode'].unique()
    mapping_dict = {}
    class_names = [] 

    for i, crop_code in enumerate(crop_codes):
        mapping_dict[crop_code] = i
        crop_type = df_sklearn[df_sklearn['afgkode'] == crop_code].head(1)['afgroede'].values[0]
        class_names.append(crop_type)

    df_sklearn_remapped = df_sklearn.copy()
    df_sklearn_remapped['afgkode'] = df_sklearn_remapped['afgkode'].map(mapping_dict)
    #print(f"Crop types: {class_names}")

    array = df_sklearn_remapped.values

    # Define the independent variables as features.
    X = np.float32(array[:,3:])  # The features 

    # Define the target (dependent) variable as labels.
    y = np.int8(array[:,1])  # The column 'afgkode'

    # Create a train/test split using 30% test size.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

    #print(f"Train samples:      {len(y_train)}")
    #print(f"Test samples:       {len(y_test)}")
    #print(f"Number of features: {len(X[0,:])}")

    from sklearn.model_selection import GridSearchCV
    param_grid = {'C': [1, 10, 100], 'gamma': [0.001, 0.01, 0.1], 'kernel': ['rbf']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5, verbose=2, n_jobs=16)

    grid_trained, _ = evaluate_classifier(grid, X_train, X_test, y_train, y_test, class_names, feature_scale=True)
```

```python
year = 2018
for i in range(7, 24, 1):
    month = (i % 12) + 1
    if month == 1:
        year += 1
        
    print(f"--------------------------------------------------------------------------------------------------")
    print(f"Dataset from 2018-07-01 to {year}-{month:02}-01")
    df_sklearn = get_sklearn_df(polygons_year=2019, 
                                satellite_dates=slice('2018-07-01', f'{year}-{month:02}-01'), 
                                fields='all', 
                                satellite='all', 
                                polarization='all',
                                crop_type='all',
                                netcdf_path=netcdf_path)

    df_sklearn = df_sklearn[df_sklearn['afgroede'].isin(['Vårbyg', 'Vinterhvede', 'Silomajs', 'Vinterraps', 
                                                         'Vinterbyg', 'Vårhavre', 'Vinterhybridrug'])]
    crop_codes = df_sklearn['afgkode'].unique()
    mapping_dict = {}
    class_names = [] 

    for i, crop_code in enumerate(crop_codes):
        mapping_dict[crop_code] = i
        crop_type = df_sklearn[df_sklearn['afgkode'] == crop_code].head(1)['afgroede'].values[0]
        class_names.append(crop_type)

    df_sklearn_remapped = df_sklearn.copy()
    df_sklearn_remapped['afgkode'] = df_sklearn_remapped['afgkode'].map(mapping_dict)
    #print(f"Crop types: {class_names}")

    array = df_sklearn_remapped.values

    # Define the independent variables as features.
    X = np.float32(array[:,3:])  # The features 

    # Define the target (dependent) variable as labels.
    y = np.int8(array[:,1])  # The column 'afgkode'

    # Create a train/test split using 30% test size.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

    #print(f"Train samples:      {len(y_train)}")
    #print(f"Test samples:       {len(y_test)}")
    #print(f"Number of features: {len(X[0,:])}")

    from sklearn.svm import SVC   
    from sklearn.model_selection import GridSearchCV
    param_grid = {'C': [1, 10, 100, 1000], 'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 'kernel': ['rbf']}
    grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, refit=True, cv=5, verbose=20, n_jobs=32)

    grid_trained, _ = evaluate_classifier(grid, X_train, X_test, y_train, y_test, class_names, feature_scale=True)
```

```python

```
