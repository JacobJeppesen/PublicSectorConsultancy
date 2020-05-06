```python
import os
import random
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns

sns.set_style('ticks')
from pathlib import Path
from tqdm.autonotebook import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split         # Split data into train and test set
from sklearn.metrics import classification_report            # Summary of classifier performance

from utils import get_df, evaluate_classifier, numpy_confusion_matrix_to_pycm, save_confusion_matrix_fig 

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
    'Vårbyg': 'Spring barley',
    'Vinterbyg': 'Winter barley',
    'Vårhvede': 'Spring wheat',
    'Vinterhvede': 'Winter wheat',
    'Vinterrug': 'Winter rye',
    'Vårhavre': 'Spring oat',
    'Silomajs': 'Maize',
    'Vinterraps': 'Rapeseed',
    'Permanent græs, normalt udbytte': 'Permanent grass',
    'Pil': 'Willow',
    'Skovdrift, alm.': 'Forest'
}

# Set global seed for random generators
RANDOM_SEED = 42

# Seed the random generators
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
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
                             satellite_dates=slice('2018-01-01', '2019-12-31'), 
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
print(f"Crop types: {class_names}")
```

```python
array = df_sklearn_remapped.values

# Define the independent variables as features.
X = np.float32(array[:,5:])  # The features 

# Define the target (dependent) variable as labels.
y = np.int8(array[:,4])  # The column 'afgkode'

# Create a train/test split using 30% test size.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, shuffle=True, random_state=RANDOM_SEED)

print(f"Train samples:      {len(y_train)}")
print(f"Test samples:       {len(y_test)}")
print(f"Number of features: {len(X[0,:])}")
```

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression          
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def get_classifiers(random_seed=42):
    # From https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    # Note: GaussianClassifier does not work (maybe requires too much training - kernel restarts in jupyter)
    N_JOBS=4
    classifiers = { 
        #'Nearest Neighbors': GridSearchCV(KNeighborsClassifier(), 
        #                                  param_grid={'n_neighbors': [2, 3, 4, 5, 6, 7, 8]}, 
        #                                  refit=True, cv=5, n_jobs=N_JOBS),
        'Decision Tree': GridSearchCV(DecisionTreeClassifier(random_state=random_seed, class_weight='balanced'), 
                                      param_grid={'max_depth': [2, 4, 6, 8, 10, 12, 14, 16]}, 
                                      refit=True, cv=5, n_jobs=N_JOBS),
        'Random Forest': GridSearchCV(RandomForestClassifier(random_state=random_seed, class_weight='balanced'), 
                                      param_grid={'max_depth': [2, 4, 6, 8, 10, 12, 14, 16], 
                                                  'n_estimators': [6, 8, 10, 12, 14], 
                                                  'max_features': [1, 2, 3]},
                                      refit=True, cv=5, n_jobs=N_JOBS),
        'Logistic Regression': GridSearchCV(LogisticRegression(max_iter=1000, random_state=random_seed, class_weight='balanced'),
                                            param_grid={'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2],
                                                        'penalty': ['none', 'l2']},
                                            refit=True, cv=5, n_jobs=N_JOBS),
        'Linear SVM': GridSearchCV(SVC(kernel='linear', random_state=random_seed, class_weight='balanced'),
                                   #param_grid={'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]},
                                   param_grid={'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]},
                                   refit=True, cv=5, n_jobs=N_JOBS),
        'RBF SVM': GridSearchCV(SVC(kernel='rbf', random_state=random_seed, class_weight='balanced'),
                                #param_grid={'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]},
                                param_grid={'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]},
                                refit=True, cv=5, n_jobs=N_JOBS),
        'Neural Network': GridSearchCV(MLPClassifier(max_iter=1000, random_state=random_seed),
                                       param_grid={'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                                                   'hidden_layer_sizes': [(50,50,50), (100, 100, 100), (100,)],
                                                   'activation': ['tanh', 'relu'],
                                                   'learning_rate': ['constant','adaptive']},
                                       refit=True, cv=5, n_jobs=N_JOBS)
        }
    
    return classifiers
```

```python
df_clf_results = pd.DataFrame(columns=['Classifier', 'Crop type', 'Prec.', 'Recall', 
                                       'F1-Score', 'Accuracy', 'Samples', 'Random seed'])

clf_trained_dict = {}
report_dict = {}
cm_dict = {}

# TODO: Also calculate uncertainties - ie. use multiple random seeds.
#       Create df (with cols [Clf_name, Random_seed, Acc., Prec., Recall, F1-score]) and loop over random seeds
#       See following on how to format pandas dataframe to get the uncertainties into the df
#       https://stackoverflow.com/questions/46584736/pandas-change-between-mean-std-and-plus-minus-notations
for random_seed in range(10):
    print(f"\n\n################################## RANDOM SEED IS SET TO {random_seed:2d} ##################################") 
    # Seed the random generators
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    
    # Get the train and test sets with the specified random_seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, shuffle=True, random_state=random_seed)
    
    # Get the classifiers
    classifiers = get_classifiers(random_seed)
    
    for name, clf in classifiers.items():
        # Evaluate classifier
        print("-------------------------------------------------------------------------------")
        print(f"Evaluating classifier: {name}")
        clf_trained, _, _, results_report, cnf_matrix = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, 
                                                                            feature_scale=True, plot_conf_matrix=False,
                                                                            print_classification_report=False)      
        print(f"The best parameters are {clf_trained.best_params_} with a score of {clf_trained.best_score_:2f}")

        # Save results in dicts
        clf_trained_dict[f'{name}_random_seed_{random_seed:02}'] = clf_trained
        report_dict[f'{name}_random_seed_{random_seed:02}'] = results_report
        cm_dict[f'{name}_random_seed_{random_seed:02}'] = cnf_matrix
        
        # Save results for individual crops in df
        df_results = pd.DataFrame(results_report).transpose()  
        for crop_type in class_names:
            # Get values
            prec = df_results.loc[crop_type, 'precision']
            recall = df_results.loc[crop_type, 'recall']
            f1 = df_results.loc[crop_type, 'f1-score']
            samples = df_results.loc[crop_type, 'support']
            acc = None

            # Insert row in df (https://stackoverflow.com/a/24284680/12045808)
            df_clf_results.loc[-1] = [name, crop_type, prec, recall, f1, acc, samples, random_seed]
            df_clf_results.index = df_clf_results.index + 1  # shifting index
            df_clf_results = df_clf_results.sort_index()  # sorting by index

        # Save overall results
        prec = df_results.loc['weighted avg', 'precision']
        recall = df_results.loc['weighted avg', 'recall']
        f1 = df_results.loc['weighted avg', 'f1-score']
        acc = df_results.loc['accuracy', 'f1-score']
        samples = df_results.loc['weighted avg', 'support']

        # Insert row in df (https://stackoverflow.com/a/24284680/12045808)
        df_clf_results.loc[-1] = [name, 'Overall', prec, recall, f1, acc, samples, random_seed]
        df_clf_results.index = df_clf_results.index + 1  # shifting index
        df_clf_results = df_clf_results.sort_index()  # sorting by index

        # Save df with results to disk
        save_path = PROJ_PATH / 'notebooks' / '03_TrainClassifiers_results.pkl'
        df_clf_results.to_pickle(save_path)
```

```python
# Load the df with results from saved file
load_path = PROJ_PATH / 'notebooks' / '03_TrainClassifiers_results.pkl'
df_clf_results = pd.read_pickle(load_path)
```

```python
df_clf_results.head(5)
```

```python
# Create df with results from classifiers and create latex table with them
df_clf_results = pd.DataFrame(columns=['Name', 'Acc.', 'Prec.', 'Recall', 'F1-Score'])
for name, results_report in report_dict.items():
    df_results = pd.DataFrame(report_dict[name]).transpose()  
    prec = df_results.loc['weighted avg', 'precision']
    recall = df_results.loc['weighted avg', 'recall']
    f1 = df_results.loc['weighted avg', 'f1-score']
    acc = df_results.loc['accuracy', 'f1-score']
    
    # Insert row in df (https://stackoverflow.com/a/24284680/12045808)
    df_clf_results.loc[-1] = [name, acc, prec, recall, f1]
    df_clf_results.index = df_clf_results.index + 1  # shifting index
    df_clf_results = df_clf_results.sort_index()  # sorting by index
    
# Print df in latex format
pd.options.display.float_format = '{:.2f}'.format  # Show 2 decimals
print(df_clf_results.sort_index(ascending=False).to_latex(index=False))  
```

```python
# Show performance on different classes with best performing classifier
# Get classfication report as pandas df
df_results = pd.DataFrame(report_dict['RBF SVM']).transpose()  

# Round the values to 2 decimals
df_results = df_results.astype({'support': 'int32'}).round(2) 

# Remove samples from 'macro avg' and 'weighed avg'
df_results.loc[df_results.index == 'accuracy', 'precision'] = ''  
df_results.loc[df_results.index == 'accuracy', 'recall'] = ''  
df_results.loc[df_results.index == 'accuracy', 'support'] = df_results.loc[df_results.index == 'macro avg', 'support'].values

# Rename the support column to 'samples'
df_results = df_results.rename(columns={'precision': 'Prec.',
                                        'recall': 'Recall',
                                        'f1-score': 'F1-score',
                                        'support': 'Samples'},
                               index={'accuracy': 'Overall acc.',
                                      'macro avg': 'Macro avg.',
                                      'weighted avg': 'Weighted avg.'})


# Print df in latex format (I normally add a /midrule above 'Macro avg.' and delete 'Overall acc.')
pd.options.display.float_format = '{:.2f}'.format  # Show 2 decimals
print(df_results.to_latex(index=True))  
```

```python
# Save the confusion matrix figure you want
plot_name = 'RBF_SVM_conf_matrix'
plot_path = PROJ_PATH / 'reports' / 'figures' / 'conf_matrices' / plot_name
if not plot_path.parent.exists():
    plot_path.parent.mkdir()
    
save_confusion_matrix_fig(cm_dict['RBF SVM'], classes=class_names, save_path=plot_path)
save_confusion_matrix_fig(cm_dict['RBF SVM'], classes=class_names, save_path=plot_path, normalized=True)
```

```python

```
