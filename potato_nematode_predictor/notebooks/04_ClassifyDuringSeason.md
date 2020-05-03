```python
import os
import random
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns

sns.set_style('ticks')
from pathlib import Path
from matplotlib import pyplot as plt
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

# Set seed for random generators
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
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression          
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# From https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# Note: GaussianClassifier does not work (maybe requires too much training - kernel restarts in jupyter)
N_JOBS=24
classifiers = { 
    'Nearest Neighbors': GridSearchCV(KNeighborsClassifier(), 
                                      param_grid={'n_neighbors': [2, 3, 4, 5, 6, 7, 8]}, 
                                      refit=True, cv=5, n_jobs=N_JOBS),
    'Decision Tree': GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_SEED, class_weight='balanced'), 
                                  param_grid={'max_depth': [2, 4, 6, 8, 10, 12, 14, 16]}, 
                                  refit=True, cv=5, n_jobs=N_JOBS),
    'Random Forest': GridSearchCV(RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced'), 
                                  param_grid={'max_depth': [2, 4, 6, 8, 10, 12, 14, 16], 
                                              'n_estimators': [6, 8, 10, 12, 14], 
                                              'max_features': [1, 2, 3]},
                                  refit=True, cv=5, n_jobs=N_JOBS),
    'Logistic Regression': GridSearchCV(LogisticRegression(random_state=RANDOM_SEED, class_weight='balanced'),
                                        param_grid={'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2],
                                                    'penalty': ['none', 'l2']},
                                        refit=True, cv=5, n_jobs=N_JOBS),
    'Linear SVM': GridSearchCV(SVC(kernel='linear', random_state=RANDOM_SEED, class_weight='balanced'),
                               #param_grid={'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]},
                               param_grid={'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]},
                               refit=True, cv=5, n_jobs=N_JOBS),
    'RBF SVM': GridSearchCV(SVC(kernel='rbf', random_state=RANDOM_SEED, class_weight='balanced'),
                            #param_grid={'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]},
                            param_grid={'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]},
                            refit=True, cv=5, n_jobs=N_JOBS),
    'Neural Network': GridSearchCV(MLPClassifier(max_iter=1000, random_state=RANDOM_SEED),
                                   param_grid={'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                                               'hidden_layer_sizes': [(50,50,50), (100, 100, 100), (100,)],
                                               'activation': ['tanh', 'relu'],
                                               'learning_rate': ['constant','adaptive']},
                                   refit=True, cv=5, n_jobs=N_JOBS)
    }


```

```python
def remap_df(df_sklearn):
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
    
    return df_sklearn_remapped, class_names
```

```python
df_clf_results = pd.DataFrame(columns=['Classifier', 'Date', 'Crop type', 'Prec.', 'Recall', 
                                       'F1-Score', 'Accuracy', 'Samples', 'Random seed'])

# Add an extra month to the date range at each iteration in the loop
year = 2018
#for i in range(7, 24, 1):
for i in range(7, 10, 1):
    month = (i % 12) + 1
    if month == 1:
        year += 1
        
    end_date = f'{year}-{month:02}-01'
        
    print(f"\n\n#########################################")
    print(f"# Dataset from 2018-07-01 to {end_date} #")
    print(f"#########################################\n")
    df_sklearn = get_sklearn_df(polygons_year=2019, 
                                satellite_dates=slice('2018-07-01', f'{end_date}'), 
                                fields='all', 
                                satellite='all', 
                                polarization='all',
                                crop_type='all',
                                netcdf_path=netcdf_path)
    
    df_sklearn_remapped, class_names = remap_df(df_sklearn)
    
    # Get values as numpy array
    array = df_sklearn_remapped.values
    X = np.float32(array[:,5:])  # The features 
    y = np.int8(array[:,4])  # The column 'afgkode'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, shuffle=True, random_state=RANDOM_SEED)
    
    # TODO: Also calculate uncertainties - ie. use multiple random seeds.
    #       Create df (with cols [Clf_name, Random_seed, Acc., Prec., Recall, F1-score]) and loop over random seeds
    #       See following on how to format pandas dataframe to get the uncertainties into the df
    #       https://stackoverflow.com/questions/46584736/pandas-change-between-mean-std-and-plus-minus-notations
    
    for name, clf in classifiers.items():
        # Evaluate classifier
        print("-------------------------------------------------------------------------------")
        print(f"Evaluating classifier: {name}")
        clf_trained, _, _, results_report, cnf_matrix = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, feature_scale=True,
                                                                           plot_conf_matrix=False, print_classification_report=True)      
        print(f"The best parameters are {clf_trained.best_params_} with a score of {clf_trained.best_score_:2f}")

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
            df_clf_results.loc[-1] = [name, end_date, crop_type, prec, recall, f1, acc, samples, RANDOM_SEED]
            df_clf_results.index = df_clf_results.index + 1  # shifting index
            df_clf_results = df_clf_results.sort_index()  # sorting by index
        
        # Save overall results
        prec = df_results.loc['weighted avg', 'precision']
        recall = df_results.loc['weighted avg', 'recall']
        f1 = df_results.loc['weighted avg', 'f1-score']
        acc = df_results.loc['accuracy', 'f1-score']
        samples = df_results.loc['weighted avg', 'support']
        
        # Insert row in df (https://stackoverflow.com/a/24284680/12045808)
        df_clf_results.loc[-1] = [name, end_date, 'Overall', prec, recall, f1, acc, samples, RANDOM_SEED]
        df_clf_results.index = df_clf_results.index + 1  # shifting index
        df_clf_results = df_clf_results.sort_index()  # sorting by index
    
```

```python
# If you loop over random seeds, the confidence interval can be made just by setting ci='std'
df_overall = df_clf_results[df_clf_results['Crop type'] == 'Overall'].astype({'Accuracy': 'float64'})
ax = sns.lineplot(x="Date", y="Accuracy", hue='Classifier', data=df_overall.sort_index(ascending=False), markers=True, ci=None)
#ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_ylabel('Test accuracy')
ax.set_xlabel('')
ax.set_ylim(0, 1)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
```

```python
df_crop = df_clf_results[df_clf_results['Crop type'] != 'Overall']
df_crop = df_crop[df_crop['Classifier'] == 'Decision Tree']

# Define markers 
# Note: Markers are not working in lineplots (see https://github.com/mwaskom/seaborn/issues/1513#issuecomment-480261748)
filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D')

ax = sns.lineplot(x="Date", y="F1-Score", hue='Crop type', data=df_crop.sort_index(ascending=False), ci=None)
ax.set_ylabel('F1-score')
ax.set_xlabel('')
ax.set_ylim(0, 1)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
    
ax.legend(bbox_to_anchor=(1.05, 0.95), loc=2, borderaxespad=0.)
#ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
```
