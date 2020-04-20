```python
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns

from pathlib import Path
from tqdm.autonotebook import tqdm
from sklearn.model_selection import train_test_split         # Split data into train and test set
from sklearn.metrics import classification_report            # Summary of classifier performance

from utils import get_df, evaluate_classifier

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

# Mapping dict
mapping_dict_crop_types = {
    'Kartofler, stivelses-': 'Potato',
    'Kartofler, lægge- (egen opformering)': 'Potato',
    'Kartofler, andre': 'Potato',
    'Kartofler, spise-': 'Potato',
    'Kartofler, lægge- (certificerede)': 'Potato',
    'Vårbyg': 'Barley',
    'Vinterbyg': 'Barley',
    'Grønkorn af vårbyg': 'Barley',
    'Vårbyg, helsæd': 'Barley',
    'Vinterhvede': 'Wheat',
    'Vårhvede': 'Wheat',
    'Vinterhybridrug': 'Rye',
    'Vårhavre': 'Oat',
    'Silomajs': 'Maize',
    'Majs til modenhed': 'Maize',
    'Vinterraps': 'Rapeseed',
    'Sukkerroer til fabrik': 'Sugarbeet',
    'Permanent græs, normalt udbytte': 'Grass',
    'Skovdrift, alm.': 'Forest',
    'Juletræer og pyntegrønt på landbrugsjord': 'Forest'
}

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
# Create the df format to be used by scikit-learn
for i, polarization in enumerate(['VV', 'VH', 'VV-VH']):
    df_polarization = get_df(polygons_year=2019, 
                             satellite_dates=slice('2019-01-01', '2019-10-01'), 
                             fields='all', 
                             satellite='all', 
                             polarization=polarization,
                             netcdf_path=netcdf_path)
    
    # Extract a mapping of field_ids to crop type
    if i == 0:
        df_sklearn = df_polarization[['field_id', 'afgkode', 'afgroede']]
    
    # Pivot the df (https://stackoverflow.com/a/37790707/12045808)
    df_polarization = df_polarization.pivot(index='field_id', columns='date', values='stats_mean')
    
    # Add polarization to column names
    df_polarization.columns = [str(col)[:10]+f'_{polarization}' for col in df_polarization.columns]  
    
    # Merge the polarization dataframes into one dataframe
    df_polarization = df_polarization.reset_index()  # Creates new indices and a 'field_id' column (field id was used as indices before)
    df_sklearn = pd.merge(df_sklearn, df_polarization, on='field_id') 
        
# Drop fields having nan values
df_sklearn = df_sklearn.dropna()

# The merge operation for some reason made duplicates (there was a bug reported on this earlier), so drop duplicates and re-index the df
df_sklearn = df_sklearn.drop_duplicates().reset_index(drop=True)
```

```python
#df_sklearn
```

```python
#df_sklearn[df_sklearn['afgkode'] == 252].describe()
```

```python
#df_sklearn = df_sklearn[df_sklearn['afgroede'].isin(['Vårbyg', 'Vinterhvede', 'Silomajs', 'Vinterraps', 
#                                                     'Vinterbyg', 'Vårhavre', 'Vinterhybridrug'])]

df_sklearn_remapped = df_sklearn.copy()

df_sklearn_remapped.insert(3, 'Crop type', '')
df_sklearn_remapped.insert(4, 'Label ID', 0)
mapping_dict = {}
class_names = [] 
i = 0
for key, value in mapping_dict_crop_types.items():
    df_sklearn_remapped.loc[df_sklearn_remapped['afgroede'] == key, 'Crop type'] = value 
    if value not in class_names:
        class_names.append(value)
        mapping_dict[value] = i
        i += 1
    
    #df_sklearn = df_sklearn.replace(key, value)
    
for key, value in mapping_dict.items():
    df_sklearn_remapped.loc[df_sklearn_remapped['Crop type'] == key, 'Label ID'] = value 
print(f"Crop types: {class_names}")
```

```python
array = df_sklearn_remapped.values

# Define the independent variables as features.
X = np.float32(array[:,5:])  # The features 

# Define the target (dependent) variable as labels.
y = np.int8(array[:,4])  # The column 'afgkode'
```

```python
# Create a train/test split using 30% test size.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

print(f"Train samples:      {len(y_train)}")
print(f"Test samples:       {len(y_test)}")
print(f"Number of features: {len(X[0,:])}")
```

```python
from sklearn.tree import DecisionTreeClassifier              

# Instantiate and evaluate classifier
clf = DecisionTreeClassifier()
clf_trained, _ = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, feature_scale=False)
```

```python
from sklearn.linear_model import LogisticRegression          

# Instantiate classifier.
#clf = LogisticRegression(solver='newton-cg', max_iter=1000)
clf = LogisticRegression()

# Evaluate classifier without feature scaling
clf_trained, _ = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, feature_scale=True)
```

```python
from sklearn.neural_network import MLPClassifier

# Instantiate and evaluate classifier
clf = MLPClassifier(solver='lbfgs', alpha=10, hidden_layer_sizes=(25, 25), max_iter=1000)  # See what happens when you change random state
clf_trained, _ = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, feature_scale=True)
```

```python
from sklearn.svm import SVC   

# Instantiate and evaluate classifier
clf = SVC(kernel='linear')
clf_trained, _ = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names,  feature_scale=True)
```

```python
# Instantiate and evaluate classifier
clf = SVC(kernel='rbf')
clf_trained, _ = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names,  feature_scale=True)
```

```python
try:  # If auto-sklearn is installed 
    import autosklearn.classification
except:  # Else install auto-sklearn (https://automl.github.io/auto-sklearn/master/installation.html and https://hub.docker.com/r/alfranz/automl/dockerfile) 
    !sudo apt-get update && sudo apt-get install -y swig curl
    !curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install --default-timeout=100
    !pip install auto-sklearn
```

```python
import autosklearn.classification

# Instantiate and evaluate classifier
#clf = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=3600, per_run_time_limit=360, 
#                                                       ml_memory_limit=4096, n_jobs=12,  resampling_strategy='cv',
#                                                       resampling_strategy_arguments={'folds': 5},)
clf = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=3600, per_run_time_limit=360, 
                                                       ml_memory_limit=32768, n_jobs=24)
clf_trained, _ = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names,  feature_scale=True)

# Then train the ensemble on the whole training dataset
# https://automl.github.io/auto-sklearn/master/examples/example_crossvalidation.html#sphx-glr-examples-example-crossvalidation-py
```

```python
import autosklearn.classification

# Instantiate and evaluate classifier
clf = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=3600, per_run_time_limit=360, 
                                                       ml_memory_limit=32768, n_jobs=24,  resampling_strategy='cv',
                                                       resampling_strategy_arguments={'folds': 10},)
#clf = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=3600, per_run_time_limit=360, 
#                                                       ml_memory_limit=32768, n_jobs=24)
clf_trained, _ = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, feature_scale=True, auto_sklearn_crossvalidation=True)

# Then train the ensemble on the whole training dataset
# https://automl.github.io/auto-sklearn/master/examples/example_crossvalidation.html#sphx-glr-examples-example-crossvalidation-py
```

```python
#clf_trained[0].refit(X_train, y_train)
```

```python
#predictions = clf.predict(X_test)
#report = classification_report(y_test, predictions, target_names=class_names)
#print(report)
```

```python
#clf.show_models()
```

```python

```
