```python
import os
import random
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns

from pathlib import Path
from tqdm.autonotebook import tqdm
from matplotlib import pyplot as plt
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, shuffle=True, random_state=RANDOM_SEED)

print(f"Train samples:      {len(y_train)}")
print(f"Test samples:       {len(y_test)}")
print(f"Number of features: {len(X[0,:])}")
```

```python
from sklearn.tree import DecisionTreeClassifier              

# Instantiate and evaluate classifier
clf = DecisionTreeClassifier()
clf_trained, _, _, _, _ = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, feature_scale=False)
```

```python
from sklearn.linear_model import LogisticRegression          

# Instantiate classifier.
#clf = LogisticRegression(solver='newton-cg', max_iter=1000)
clf = LogisticRegression()

# Evaluate classifier without feature scaling
clf_trained, _, _, _ = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, feature_scale=True)
```

```python
from sklearn.neural_network import MLPClassifier

# Instantiate and evaluate classifier
clf = MLPClassifier(solver='lbfgs', alpha=10, hidden_layer_sizes=(25, 25), max_iter=1000)  # See what happens when you change random state
clf_trained, _, _, _ = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, feature_scale=True)
```

```python
from sklearn.svm import SVC   

# Instantiate and evaluate classifier
clf = SVC(kernel='linear')
clf_trained, _, _, _ = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names,  feature_scale=True)
```

```python
# Instantiate and evaluate classifier
clf = SVC(kernel='rbf')
clf_trained, _, _, _ = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, feature_scale=True)
```

```python
# Instantiate and evaluate classifier
clf = SVC(kernel='rbf', class_weight='balanced')
clf_trained, _, _, _ = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, feature_scale=True)
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
clf = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=7200, per_run_time_limit=720, 
                                                       ml_memory_limit=32768, n_jobs=12,  resampling_strategy='cv',
                                                       resampling_strategy_arguments={'folds': 5},)
#clf = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=3600, per_run_time_limit=360, 
#                                                       ml_memory_limit=32768, n_jobs=24)
clf_trained, _, _, _ = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, feature_scale=True, auto_sklearn_crossvalidation=True)

# Then train the ensemble on the whole training dataset
# https://automl.github.io/auto-sklearn/master/examples/example_crossvalidation.html#sphx-glr-examples-example-crossvalidation-py
```

```python
# Take a look at https://towardsdatascience.com/svm-hyper-parameter-tuning-using-gridsearchcv-49c0bc55ce29
from sklearn.svm import SVC   
from sklearn.model_selection import GridSearchCV
#param_grid = {'C': [1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], 
#              'gamma': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], 
#              'kernel': ['rbf']}
param_grid = {'C': [1e1], 
              'gamma': [1e-3], 
              'kernel': ['rbf']}
#grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, refit=True, cv=5, verbose=20, n_jobs=32)
grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5, verbose=20, n_jobs=16)
grid_trained, _, _, results_report, cnf_matrix = evaluate_classifier(grid, X_train, X_test, y_train, y_test, class_names, feature_scale=True)

print(f"The best parameters are {grid_trained.best_params_} with a score of {grid_trained.best_score_:2f}")
```

```python
# Idea: Maybe make a utils folder, with a plotting module, evaluation module etc.. The below here should 
#       then be put in the plotting module. 
mean_test_scores = grid_trained.cv_results_['mean_test_score']
mean_fit_times = grid_trained.cv_results_['mean_fit_time']
param_columns = list(grid_trained.cv_results_['params'][0].keys())
result_columns = ['mean_fit_time', 'mean_test_score']
num_fits = len(grid_trained.cv_results_['params'])

df_cv_results = pd.DataFrame(0, index=range(num_fits), columns=param_columns+result_columns)
for i, param_set in enumerate(grid_trained.cv_results_['params']):
    for param, value in param_set.items():
        df_cv_results.loc[i, param] = value 
    df_cv_results.loc[i, 'mean_test_score'] = mean_test_scores[i]
    df_cv_results.loc[i, 'mean_fit_time'] = mean_fit_times[i]
    
df_heatmap_mean_score = df_cv_results.pivot(index='C', columns='gamma', values='mean_test_score')
plt.figure(figsize=(10,8))
ax = sns.heatmap(df_heatmap_mean_score, annot=True, cmap=plt.cm.Blues)

df_heatmap_fit_time = df_cv_results.pivot(index='C', columns='gamma', values='mean_fit_time')
plt.figure(figsize=(10,8))
ax = sns.heatmap(df_heatmap_fit_time.astype('int64'), annot=True, fmt='d', cmap=plt.cm.Blues_r)
```

```python
# Get classfication report as pandas df
df_results = pd.DataFrame(results_report).transpose()  

# THIS IS WRONG! IT IS RECALL WE ARE CALCULATING!
# Get class-wise accuracy from confusion matrix
#class_wise_acc = cnf_matrix.diagonal()/cnf_matrix.sum(axis=1)

# Get macro avg of accuracies
#macro_avg_acc = np.average(class_wise_acc)

# Get weighted avg of accuracies
#samples = df_results.support.values[:-3]  # Bottom 3 rows are acc., macro avg and weighted avg in report
#weights = samples / np.sum(samples)
#weighted_avg_acc = np.average(class_wise_acc, weights=weights)

# Drop the accuracies row
#df_results = df_results[df_results.index != 'accuracy']

# Add accuracies column
#acc_column_values = np.append(class_wise_acc, [macro_avg_acc, weighted_avg_acc])
#df_results.insert (0, 'Acc.', acc_column_values)

# Round the values to 2 decimals
df_results = df_results.astype({'support': 'int32'}).round(2) 

# Remove samples from 'macro avg' and 'weighed avg'
df_results.loc[df_results.index == 'accuracy', 'precision'] = ''  
df_results.loc[df_results.index == 'accuracy', 'recall'] = ''  
df_results.loc[df_results.index == 'accuracy', 'support'] = df_results.loc[df_results.index == 'macro avg', 'support'].values
#df_results.loc[df_results.index == 'weighted avg', 'support'] = ''  
#df_results.loc[df_results.index == 'macro avg', 'support'] = ''  

# Rename the support column to 'samples'
df_results = df_results.rename(columns={'precision': 'Prec.',
                                        'recall': 'Recall',
                                        'f1-score': 'F1-score',
                                        'support': 'Samples'},
                               index={'accuracy': 'Overall acc.',
                                      'macro avg': 'Macro avg.',
                                      'weighted avg': 'Weighted avg.'})


# Print df in latex format (I normally add a /midrule above 'Macro avg.')
pd.options.display.float_format = '{:.2f}'.format  # Show 2 decimals
print(df_results.to_latex(index=True))  
```

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# From https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# Note: GaussianClassifier does not work (maybe requires too much training - kernel restarts in jupyter)
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=10),
    SVC(gamma=0.001, C=10),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

# TODO: Also calculate uncertainties - ie. use multiple random seeds.
#       Create df (with cols [Clf_name, Random_seed, Acc., Prec., Recall, F1-score]) and loop over random seeds
#       See following on how to format pandas dataframe to get the uncertainties into the df
#       https://stackoverflow.com/questions/46584736/pandas-change-between-mean-std-and-plus-minus-notations

for name, clf in zip(names, classifiers):
    print("-------------------------------------------------------------------------------")
    print(f"Evaluating classifier: {name}")
    clf_trained, _, _, results_report, cnf_matrix = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, feature_scale=True)      
```

```python
try:
    from pycm import ConfusionMatrix
except:
    !pip install pycm
    from pycm import ConfusionMatrix

def numpy_confusion_matrix_to_pycm(confusion_matrix_numpy, labels=None):
    """Create a pycm confusion matrix from a NumPy confusion matrix
    Creates a confusion matrix object with the pycm library based on a confusion matrix as 2D NumPy array (such as
    the one generated by the sklearn confusion matrix function).
    See more about pycm confusion matrices at `pycm`_, and see more
    about sklearn confusion matrices at `sklearn confusion matrix`_.
    Args:
        confusion_matrix_numpy (np.array((num_classes, num_classes)) :
        labels (list) :
    Returns:
        confusion_matrix_pycm (pycm.ConfusionMatrix) :
    .. _`pycm`: https://github.com/sepandhaghighi/pycm
    .. _`sklearn confusion matrix`:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    # Create empty dict to be used as input for pycm (see https://github.com/sepandhaghighi/pycm#direct-cm)
    confusion_matrix_dict = {}

    # Find number and classes and check labels
    num_classes = np.shape(confusion_matrix_numpy)[0]
    if not labels:  # If no labels are provided just use [0, 1, ..., num_classes]
        labels = range(num_classes)
    elif len(labels) != num_classes:
        raise AttributeError("Number of provided labels does not match number of classes.")

    # Fill the dict in the format required by pycm with values from the sklearn confusion matrix
    for row in range(num_classes):
        row_dict = {}
        for col in range(num_classes):
            row_dict[str(labels[col])] = int(confusion_matrix_numpy[row, col])
        confusion_matrix_dict[str(labels[row])] = row_dict

    # Instantiate the pycm confusion matrix from the dict
    confusion_matrix_pycm = ConfusionMatrix(matrix=confusion_matrix_dict)

    return confusion_matrix_pycm
```

```python
pycm_confusion_matrix = numpy_confusion_matrix_to_pycm(cnf_matrix, labels=class_names)
print(pycm_confusion_matrix.ACC)
```

```python
pycm_confusion_matrix.stat()
```
