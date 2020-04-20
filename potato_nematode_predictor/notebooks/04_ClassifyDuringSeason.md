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
# Dicts to hold results
test_acc_logistic_regression = {'2018-07-01': 0}
classification_reports_logistic_regression = {}
trained_classifiers_logistic_regression = {}

year = 2018
for i in range(7, 24, 1):
    month = (i % 12) + 1
    if month == 1:
        year += 1
        
    end_date = f'{year}-{month:02}-01'
        
    print(f"--------------------------------------------------------------------------------------------------")
    print(f"Dataset from 2018-07-01 to {end_date}")
    df_sklearn = get_sklearn_df(polygons_year=2019, 
                                satellite_dates=slice('2018-07-01', f'{end_date}'), 
                                fields='all', 
                                satellite='all', 
                                polarization='all',
                                crop_type='all',
                                netcdf_path=netcdf_path)

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

    for key, value in mapping_dict.items():
        df_sklearn_remapped.loc[df_sklearn_remapped['Crop type'] == key, 'Label ID'] = value 
    #print(f"Crop types: {class_names}")

    # Get values as numpy array
    array = df_sklearn_remapped.values

    # Define the independent variables as features.
    X = np.float32(array[:,5:])  # The features 

    # Define the target (dependent) variable as labels.
    y = np.int8(array[:,4])  # The column 'afgkode'

    # Create a train/test split using 30% test size.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

    # Instantiate and evaluate classifier
    from sklearn.linear_model import LogisticRegression          
    clf = LogisticRegression(solver='lbfgs', multi_class='auto', n_jobs=32, max_iter=1000)
    clf_trained, _, accuracy_test, results_report = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, 
                                                                        feature_scale=True, plot_confusion_matrix=False,
                                                                        print_classification_report=False)
    
    test_acc_logistic_regression[end_date] = accuracy_test
    classification_reports_logistic_regression[end_date] = results_report 
    trained_classifiers_logistic_regression[end_date] = clf_trained 
```

```python
x = list(test_acc_logistic_regression.keys())
y = list(test_acc_logistic_regression.values())
ax = sns.lineplot(x=x, y=y, sort=False, lw=1)
ax.set_ylabel('Test accuracy')
ax.set_ylim(0, 1)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
```

```python
#for date, report in classification_reports_logistic_regression.items():
#    print(date)
#    print(report)
```

```python
# Dicts to hold results
test_acc_logistic_regression_balanced = {'2018-07-01': 0}
classification_reports_logistic_regression_balanced = {}
trained_classifiers_logistic_regression_balanced = {}

year = 2018
for i in range(7, 24, 1):
    month = (i % 12) + 1
    if month == 1:
        year += 1
        
    end_date = f'{year}-{month:02}-01'
        
    print(f"--------------------------------------------------------------------------------------------------")
    print(f"Dataset from 2018-07-01 to {end_date}")
    df_sklearn = get_sklearn_df(polygons_year=2019, 
                                satellite_dates=slice('2018-07-01', f'{end_date}'), 
                                fields='all', 
                                satellite='all', 
                                polarization='all',
                                crop_type='all',
                                netcdf_path=netcdf_path)

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

    for key, value in mapping_dict.items():
        df_sklearn_remapped.loc[df_sklearn_remapped['Crop type'] == key, 'Label ID'] = value 
    #print(f"Crop types: {class_names}")

    # Get values as numpy array
    array = df_sklearn_remapped.values

    # Define the independent variables as features.
    X = np.float32(array[:,5:])  # The features 

    # Define the target (dependent) variable as labels.
    y = np.int8(array[:,4])  # The column 'afgkode'

    # Create a train/test split using 30% test size.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

    # Instantiate and evaluate classifier
    from sklearn.linear_model import LogisticRegression          
    clf = LogisticRegression(solver='lbfgs', multi_class='auto', n_jobs=32, max_iter=1000, class_weight='balanced')
    clf_trained, _, accuracy_test, results_report = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, 
                                                                        feature_scale=True, plot_confusion_matrix=False,
                                                                        print_classification_report=False)
    
    test_acc_logistic_regression_balanced[end_date] = accuracy_test
    classification_reports_logistic_regression_balanced[end_date] = results_report 
    trained_classifiers_logistic_regression_balanced[end_date] = clf_trained 
```

```python
x = list(test_acc_logistic_regression_balanced.keys())
y = list(test_acc_logistic_regression_balanced.values())
ax = sns.lineplot(x=x, y=y, sort=False, lw=1)
ax.set_ylabel('Test accuracy')
ax.set_ylim(0, 1)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
```

```python
#for date, report in classification_reports_logistic_regression.items():
#    print(date)
#    print(report)
```

```python
# Dicts to hold results
test_acc_logistic_regression_cv = {'2018-07-01': 0}
classification_reports_logistic_regression_cv = {}
trained_classifiers_logistic_regression_cv = {}

year = 2018
for i in range(7, 24, 1):
    month = (i % 12) + 1
    if month == 1:
        year += 1
        
    end_date = f'{year}-{month:02}-01'
        
    print(f"--------------------------------------------------------------------------------------------------")
    print(f"Dataset from 2018-07-01 to {end_date}")
    df_sklearn = get_sklearn_df(polygons_year=2019, 
                                satellite_dates=slice('2018-07-01', f'{end_date}'), 
                                fields='all', 
                                satellite='all', 
                                polarization='all',
                                crop_type='all',
                                netcdf_path=netcdf_path)

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

    for key, value in mapping_dict.items():
        df_sklearn_remapped.loc[df_sklearn_remapped['Crop type'] == key, 'Label ID'] = value 
    #print(f"Crop types: {class_names}")

    # Get values as numpy array
    array = df_sklearn_remapped.values

    # Define the independent variables as features.
    X = np.float32(array[:,5:])  # The features 

    # Define the target (dependent) variable as labels.
    y = np.int8(array[:,4])  # The column 'afgkode'

    # Create a train/test split using 30% test size.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

    # Instantiate and evaluate classifier
    from sklearn.linear_model import LogisticRegressionCV          

    # Instantiate and evaluate classifier
    clf = LogisticRegressionCV(solver='lbfgs', multi_class='auto', cv=5, n_jobs=32, random_state=RANDOM_SEED, max_iter=1000)
    clf_trained, _, accuracy_test, results_report = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, 
                                                                        feature_scale=True, plot_confusion_matrix=False,
                                                                        print_classification_report=False)
    
    test_acc_logistic_regression_cv[end_date] = accuracy_test
    classification_reports_logistic_regression_cv[end_date] = results_report 
    trained_classifiers_logistic_regression_cv[end_date] = clf_trained 
    
```

```python
x = list(test_acc_logistic_regression_cv.keys())
y = list(test_acc_logistic_regression_cv.values())
ax = sns.lineplot(x=x, y=y, sort=False, lw=1)
ax.set_ylabel('Test accuracy')
ax.set_ylim(0, 1)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
```

```python
#for date, report in classification_reports_logistic_regression.items():
#    print(date)
#    print(report)
```

```python
# Dicts to hold results
test_acc_logistic_regression_cv_balanced = {'2018-07-01': 0}
classification_reports_logistic_regression_cv_balanced = {}
trained_classifiers_logistic_regression_cv_balanced = {}

year = 2018
for i in range(7, 24, 1):
    month = (i % 12) + 1
    if month == 1:
        year += 1
        
    end_date = f'{year}-{month:02}-01'
        
    print(f"--------------------------------------------------------------------------------------------------")
    print(f"Dataset from 2018-07-01 to {end_date}")
    df_sklearn = get_sklearn_df(polygons_year=2019, 
                                satellite_dates=slice('2018-07-01', f'{end_date}'), 
                                fields='all', 
                                satellite='all', 
                                polarization='all',
                                crop_type='all',
                                netcdf_path=netcdf_path)

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

    for key, value in mapping_dict.items():
        df_sklearn_remapped.loc[df_sklearn_remapped['Crop type'] == key, 'Label ID'] = value 
    #print(f"Crop types: {class_names}")

    # Get values as numpy array
    array = df_sklearn_remapped.values

    # Define the independent variables as features.
    X = np.float32(array[:,5:])  # The features 

    # Define the target (dependent) variable as labels.
    y = np.int8(array[:,4])  # The column 'afgkode'

    # Create a train/test split using 30% test size.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

    # Instantiate and evaluate classifier
    from sklearn.linear_model import LogisticRegressionCV          

    # Instantiate and evaluate classifier
    clf = LogisticRegressionCV(solver='lbfgs', multi_class='auto', cv=5, n_jobs=32, random_state=RANDOM_SEED, max_iter=1000, 
                               class_weight='balanced')
    clf_trained, _, accuracy_test, results_report = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, 
                                                                        feature_scale=True, plot_confusion_matrix=False,
                                                                        print_classification_report=False)
    
    test_acc_logistic_regression_cv_balanced[end_date] = accuracy_test
    classification_reports_logistic_regression_cv_balanced[end_date] = results_report 
    trained_classifiers_logistic_regression_cv_balanced[end_date] = clf_trained 
```

```python
x = list(test_acc_logistic_regression_cv_balanced.keys())
y = list(test_acc_logistic_regression_cv_balanced.values())
ax = sns.lineplot(x=x, y=y, sort=False, lw=1)
ax.set_ylabel('Test accuracy')
ax.set_ylim(0, 1)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
```

```python
#for date, report in classification_reports_logistic_regression.items():
#    print(date)
#    print(report)
```

```python
# Dicts to hold results
test_acc_svm_linear = {'2018-07-01': 0}
classification_reports_svm_linear = {}
trained_classifiers_svm_linear = {}

year = 2018
for i in range(7, 24, 1):
    month = (i % 12) + 1
    if month == 1:
        year += 1
        
    end_date = f'{year}-{month:02}-01'
        
    print(f"--------------------------------------------------------------------------------------------------")
    print(f"Dataset from 2018-07-01 to {end_date}")
    df_sklearn = get_sklearn_df(polygons_year=2019, 
                                satellite_dates=slice('2018-07-01', f'{end_date}'), 
                                fields='all', 
                                satellite='all', 
                                polarization='all',
                                crop_type='all',
                                netcdf_path=netcdf_path)

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

    for key, value in mapping_dict.items():
        df_sklearn_remapped.loc[df_sklearn_remapped['Crop type'] == key, 'Label ID'] = value 
    #print(f"Crop types: {class_names}")

    # Get values as numpy array
    array = df_sklearn_remapped.values

    # Define the independent variables as features.
    X = np.float32(array[:,5:])  # The features 

    # Define the target (dependent) variable as labels.
    y = np.int8(array[:,4])  # The column 'afgkode'

    # Create a train/test split using 30% test size.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

    # Instantiate and evaluate classifier
    from sklearn.svm import SVC   
    from sklearn.model_selection import GridSearchCV

    # Instantiate and evaluate classifier
    param_grid = {'C': [1, 10, 100], 'gamma': [0.0001, 0.001, 0.01, 0.1], 'kernel': ['linear']}
    #clf = GridSearchCV(SVC(), param_grid, refit=True, cv=5, verbose=0, n_jobs=32)
    clf = SVC(kernel='linear')
    clf_trained, _, accuracy_test, results_report = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, 
                                                                        feature_scale=True, plot_confusion_matrix=False,
                                                                        print_classification_report=False)
    
    test_acc_svm_linear[end_date] = accuracy_test
    classification_reports_svm_linear[end_date] = results_report 
    trained_classifiers_svm_linear[end_date] = clf_trained 
```

```python
x = list(test_acc_svm_linear.keys())
y = list(test_acc_svm_linear.values())
ax = sns.lineplot(x=x, y=y, sort=False, lw=1)
ax.set_ylabel('Test accuracy')
ax.set_ylim(0, 1)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
```

```python
#for date, report in classification_reports_logistic_regression.items():
#    print(date)
#    print(report)
```

```python
# Dicts to hold results
test_acc_svm_linear_balanced = {'2018-07-01': 0}
classification_reports_svm_linear_balanced = {}
trained_classifiers_svm_linear_balanced = {}

year = 2018
for i in range(7, 24, 1):
    month = (i % 12) + 1
    if month == 1:
        year += 1
        
    end_date = f'{year}-{month:02}-01'
        
    print(f"--------------------------------------------------------------------------------------------------")
    print(f"Dataset from 2018-07-01 to {end_date}")
    df_sklearn = get_sklearn_df(polygons_year=2019, 
                                satellite_dates=slice('2018-07-01', f'{end_date}'), 
                                fields='all', 
                                satellite='all', 
                                polarization='all',
                                crop_type='all',
                                netcdf_path=netcdf_path)

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

    for key, value in mapping_dict.items():
        df_sklearn_remapped.loc[df_sklearn_remapped['Crop type'] == key, 'Label ID'] = value 
    #print(f"Crop types: {class_names}")

    # Get values as numpy array
    array = df_sklearn_remapped.values

    # Define the independent variables as features.
    X = np.float32(array[:,5:])  # The features 

    # Define the target (dependent) variable as labels.
    y = np.int8(array[:,4])  # The column 'afgkode'

    # Create a train/test split using 30% test size.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

    # Instantiate and evaluate classifier
    from sklearn.svm import SVC   
    from sklearn.model_selection import GridSearchCV

    # Instantiate and evaluate classifier
    param_grid = {'C': [1, 10, 100], 'gamma': [0.0001, 0.001, 0.01, 0.1], 'kernel': ['linear']}
    #clf = GridSearchCV(SVC(class_weight='balanced'), param_grid, refit=True, cv=5, verbose=0, n_jobs=32)
    clf = SVC(kernel='linear', class_weight='balanced')
    clf_trained, _, accuracy_test, results_report = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, 
                                                                        feature_scale=True, plot_confusion_matrix=False,
                                                                        print_classification_report=False)
    
    test_acc_svm_linear_balanced[end_date] = accuracy_test
    classification_reports_svm_linear_balanced[end_date] = results_report 
    trained_classifiers_svm_linear_balanced[end_date] = clf_trained 
```

```python
x = list(test_acc_svm_linear_balanced.keys())
y = list(test_acc_svm_linear_balanced.values())
ax = sns.lineplot(x=x, y=y, sort=False, lw=1)
ax.set_ylabel('Test accuracy')
ax.set_ylim(0, 1)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
```

```python
#for date, report in classification_reports_logistic_regression.items():
#    print(date)
#    print(report)
```

```python
# Dicts to hold results
test_acc_svm_rbf = {'2018-07-01': 0}
classification_reports_svm_rbf = {}
trained_classifiers_svm_rbf = {}

year = 2018
for i in range(7, 24, 1):
    month = (i % 12) + 1
    if month == 1:
        year += 1
        
    end_date = f'{year}-{month:02}-01'
        
    print(f"--------------------------------------------------------------------------------------------------")
    print(f"Dataset from 2018-07-01 to {end_date}")
    df_sklearn = get_sklearn_df(polygons_year=2019, 
                                satellite_dates=slice('2018-07-01', f'{end_date}'), 
                                fields='all', 
                                satellite='all', 
                                polarization='all',
                                crop_type='all',
                                netcdf_path=netcdf_path)

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

    for key, value in mapping_dict.items():
        df_sklearn_remapped.loc[df_sklearn_remapped['Crop type'] == key, 'Label ID'] = value 
    #print(f"Crop types: {class_names}")

    # Get values as numpy array
    array = df_sklearn_remapped.values

    # Define the independent variables as features.
    X = np.float32(array[:,5:])  # The features 

    # Define the target (dependent) variable as labels.
    y = np.int8(array[:,4])  # The column 'afgkode'

    # Create a train/test split using 30% test size.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

    # Instantiate and evaluate classifier
    from sklearn.svm import SVC   
    from sklearn.model_selection import GridSearchCV

    # Instantiate and evaluate classifier
    param_grid = {'C': [1, 10, 100], 'gamma': [0.0001, 0.001, 0.01, 0.1], 'kernel': ['rbf']}
    #clf = GridSearchCV(SVC(), param_grid, refit=True, cv=5, verbose=0, n_jobs=32)
    clf = SVC(kernel='rbf')
    clf_trained, _, accuracy_test, results_report = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, 
                                                                        feature_scale=True, plot_confusion_matrix=False,
                                                                        print_classification_report=False)
    
    test_acc_svm_rbf[end_date] = accuracy_test
    classification_reports_svm_rbf[end_date] = results_report 
    trained_classifiers_svm_rbf[end_date] = clf_trained 
```

```python
x = list(test_acc_svm_rbf.keys())
y = list(test_acc_svm_rbf.values())
ax = sns.lineplot(x=x, y=y, sort=False, lw=1)
ax.set_ylabel('Test accuracy')
ax.set_ylim(0, 1)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
```

```python
#for date, report in classification_reports_logistic_regression.items():
#    print(date)
#    print(report)
```

```python
# Dicts to hold results
test_acc_svm_rbf_balanced = {'2018-07-01': 0}
classification_reports_svm_rbf_balanced = {}
trained_classifiers_svm_rbf_balanced = {}

year = 2018
for i in range(7, 24, 1):
    month = (i % 12) + 1
    if month == 1:
        year += 1
        
    end_date = f'{year}-{month:02}-01'
        
    print(f"--------------------------------------------------------------------------------------------------")
    print(f"Dataset from 2018-07-01 to {end_date}")
    df_sklearn = get_sklearn_df(polygons_year=2019, 
                                satellite_dates=slice('2018-07-01', f'{end_date}'), 
                                fields='all', 
                                satellite='all', 
                                polarization='all',
                                crop_type='all',
                                netcdf_path=netcdf_path)

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

    for key, value in mapping_dict.items():
        df_sklearn_remapped.loc[df_sklearn_remapped['Crop type'] == key, 'Label ID'] = value 
    #print(f"Crop types: {class_names}")

    # Get values as numpy array
    array = df_sklearn_remapped.values

    # Define the independent variables as features.
    X = np.float32(array[:,5:])  # The features 

    # Define the target (dependent) variable as labels.
    y = np.int8(array[:,4])  # The column 'afgkode'

    # Create a train/test split using 30% test size.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

    # Instantiate and evaluate classifier
    from sklearn.svm import SVC   
    from sklearn.model_selection import GridSearchCV

    # Instantiate and evaluate classifier
    param_grid = {'C': [1, 10, 100], 'gamma': [0.0001, 0.001, 0.01, 0.1], 'kernel': ['rbf']}
    #clf = GridSearchCV(SVC(class_weight='balanced'), param_grid, refit=True, cv=5, verbose=0, n_jobs=32)
    clf = SVC(kernel='rbf', class_weight='balanced')
    clf_trained, _, accuracy_test, results_report = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, 
                                                                        feature_scale=True, plot_confusion_matrix=False,
                                                                        print_classification_report=False)
    
    test_acc_svm_rbf_balanced[end_date] = accuracy_test
    classification_reports_svm_rbf_balanced[end_date] = results_report 
    trained_classifiers_svm_rbf_balanced[end_date] = clf_trained 
```

```python
x = list(test_acc_svm_rbf_balanced.keys())
y = list(test_acc_svm_rbf_balanced.values())
ax = sns.lineplot(x=x, y=y, sort=False, lw=1)
ax.set_ylabel('Test accuracy')
ax.set_ylim(0, 1)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
```

```python
#for date, report in classification_reports_logistic_regression.items():
#    print(date)
#    print(report)
```

```python

```
