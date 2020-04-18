```python
import numpy as np
import xarray as xr
import seaborn as sns

from pathlib import Path
from time import time
from tqdm.autonotebook import tqdm
from sklearn.preprocessing import StandardScaler             # Feature scaling
from sklearn.model_selection import train_test_split         # Split data into train and test set
from sklearn.metrics import classification_report            # Summary of classifier performance
from sklearn.metrics import confusion_matrix                 # Confusion matrix
from sklearn.metrics import accuracy_score

from utils import get_df

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
# Create the NumPy arrays to be used by scikit-learn. Start by finding number of fields, dates, and polarizations.
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
# NOTE: This is a quite inefficient implementation, but it is simple and easy to understand
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
for i, polarization in enumerate(['VV', 'VH', 'VV-VH']):
    df_polarization = get_df(polygons_year=2019, 
                             satellite_dates=slice('2018-01-01', '2018-12-31'), 
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
        
# Drop fields having any date with a nan value, and pick num_fields from the remainder
df_sklearn = df_sklearn.dropna()
df_sklearn = df_sklearn.drop_duplicates().reset_index(drop=True)
```

```python
df_sklearn
```

```python
df
```

```python
df_sklearn[df_cancer['diagnosis'] == "B"].describe()
```

```python
df_new = pd.merge(df_crop_types, df, on='field_id')  # The field_ids are the indices
df_new.drop_duplicates().reset_index(drop=True)
```

```python
# Create a train/test split using 30% test size.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
class_names = df['afgroede'].unique()
```

```python
def evaluate_classifer(clf, X_train, X_test, y_train, y_test, class_names, feature_scale=False):
    """
    This function evaluates a classifier. It measures training and prediction time, and 
    prints performance metrics and a confustion matrix. The returned classifier and 
    scaler are fitted to the training data, and can be used to predict new samples.
    """
    
    # Perform feature scaling
    scaler = StandardScaler()  # Scale to mean = 0 and std_dev = 1
    if feature_scale:
        # ====================== YOUR CODE HERE =======================

        X_train = scaler.fit_transform(X_train)  # Fit to training data and then scale training data
        X_test = scaler.transform(X_test)  # Scale test data based on the scaler fitted to the training data

        # =============================================================
        
    # Store the time so we can calculate training time later
    t0 = time()

    # Fit the classifier on the training features and labels
    # ====================== YOUR CODE HERE =======================

    clf.fit(X_train, y_train)

    # =============================================================
    
    # Calculate and print training time
    print("Training time:", round(time()-t0, 4), "s")

    # Store the time so we can calculate prediction time later
    t1 = time()
    
    # Use the trained classifier to classify the test data
    # ====================== YOUR CODE HERE =======================

    predictions = clf.predict(X_test)

    # =============================================================
    
    # Calculate and print prediction time
    print("Prediction time:", round(time()-t1, 4), "s")

    # Evaluate the model
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    report = classification_report(y_test, predictions, target_names=class_names)

    # Print the reports
    print("\nReport:\n")
    print("Train accuracy: {}".format(round(train_accuracy, 4)))
    print("Test accuracy: {}".format(round(test_accuracy, 4)))
    print("\n", report)
    
    # Plot confusion matrices
    cnf_matrix = confusion_matrix(y_test, predictions)
    plot_confusion_matrix(cnf_matrix, classes=class_names)
    
    # Return the trained classifier to be used on future predictions
    return clf, scaler
```

```python
from sklearn.linear_model import LogisticRegression          

# Instantiate classifier.
clf = LogisticRegression(solver='newton-cg')

# Evaluate classifier without feature scaling
#clf_trained, _ = evaluate_classifer(clf, X_train, X_test, y_train, y_test, class_names, feature_scale=False)
print(np.max(X))
```

```python
df = get_df(polygons_year=2019, 
            satellite_dates=slice('2019-01-01', '2019-03-31'), 
            fields='all', 
            satellite='all', 
            polarization='VV',
            netcdf_path=netcdf_path)
df
```

```python
for i, polarization in enumerate(['VV', 'VH', 'VV-VH']):
    df_polarization = get_df(polygons_year=2019, 
                             satellite_dates=slice('2019-01-01', '2019-01-15'), 
                             fields='all', 
                             satellite='all', 
                             polarization=polarization,
                             netcdf_path=netcdf_path)
    
    # Pivot the df (https://stackoverflow.com/a/37790707/12045808)
    df_polarization = df_polarization.pivot(index='field_id', columns='date', values='stats_mean')
    
    # Merge the polarization dataframes into one dataframe
    df_polarization.columns = [str(col)[:10]+f'_{polarization}' for col in df_polarization.columns]  # Add polarization to column names
    if i == 0:
        df = df_polarization
    else:
        df = df.merge(df_polarization, left_index=True, right_index=True)  # The field_ids are the indices
        
# Drop fields having any date with a nan value, and pick num_fields from the remainder
df = df.dropna()
```

```python
df.index
```

```python

```
