https://eli5.readthedocs.io/en/latest/_modules/eli5/sklearn/explain_weights.html
https://eli5.readthedocs.io/en/latest/tutorials/xgboost-titanic.html#explaining-predictions

Also take a look at https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html.

```python
!pip install shap
!pip install lime
```

```python
import os
import random
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
# import dabl
import shap
import lime
import lime.lime_tabular
sns.set_style('ticks')

from pathlib import Path
from tqdm.autonotebook import tqdm
from sklearn.model_selection import train_test_split         # Split data into train and test set
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from utils import evaluate_classifier, get_sklearn_df 

# Allow more rows to be printed to investigate feature importance
pd.set_option('display.max_rows', 300)

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

# Crop types to be used for explainability
explainability_crop_types = {
    'Potato': True,
    'Spring barley': False,
    'Winter barley': False,
    'Spring wheat': True,
    'Winter wheat': True,
    'Winter rye': False,
    'Spring oat': False,
    'Maize': False,
    'Rapeseed': True,
    'Permanent grass': False,
    'Willow': False,
    'Forest': True
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
df_sklearn = get_sklearn_df(polygons_year=2019, 
                            satellite_dates=slice('2019-03-01', '2019-10-01'), 
                            fields='all', 
                            satellite='S1A', 
                            polarization='all',
                            crop_type='all',
                            netcdf_path=netcdf_path)
    
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

# Drop the columns not ud for explainability
for key, value in explainability_crop_types.items():
    if not value:
        df_sklearn_remapped = df_sklearn_remapped[df_sklearn_remapped['Crop type'] != key]
print(f"Crop types used for explainability:  {df_sklearn_remapped['Crop type'].unique()}")
class_names = df_sklearn_remapped['Crop type'].unique()

# Get values as numpy array
array = df_sklearn_remapped.values

# Define the independent variables as features.
X = np.float32(array[:,5:])  # The features 

# Define the target (dependent) variable as labels.
y = np.int8(array[:,4])  # The column 'afgkode'

# Drop every n'th feature
#n = 6
#X = X[:, ::n]

# Create a train/test split using 30% test size.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
```

```python
df_plot = df_sklearn_remapped.drop(['field_id', 'afgkode', 'afgroede', 'Label ID'], axis=1)
df_plot
```

```python
df_plot['Crop type'].value_counts()
```

```python
df_plot_balanced = pd.DataFrame(columns=df_plot.columns)
for crop_type in class_names:
    df_plot_balanced = pd.concat([df_plot_balanced, df_plot[df_plot['Crop type'] == crop_type].sample(100)])
df_plot_balanced['Crop type'].value_counts()
```

```python
g = sns.PairGrid(df_plot_balanced, hue='Crop type')
g.map_diag(sns.kdeplot)
g.map_lower(plt.scatter)
g.map_upper(sns.kdeplot)
g.add_legend()
```

```python
plt.figure(figsize=(20, 20))
sns.heatmap(df_plot.corr(), annot=True)
```

```python
# Instantiate and evaluate classifier
from sklearn.linear_model import LogisticRegression          
clf = LogisticRegression(solver='lbfgs', multi_class='auto', n_jobs=32, max_iter=1000)
clf_trained, _, accuracy_test, results_report, conf_matrix = evaluate_classifier(
    clf, X_train, X_test, y_train, y_test, class_names, feature_scale=False, plot_conf_matrix=True,
    print_classification_report=True)
```

```python
# https://slundberg.github.io/shap/notebooks/linear_explainer/Sentiment%20Analysis%20with%20Logistic%20Regression.html
shap.initjs()
logistic_regression_explainer = shap.LinearExplainer(clf_trained, X_test, feature_perturbation='interventional')
logistic_regression_shap_values = logistic_regression_explainer.shap_values(X_test)
shap.summary_plot(logistic_regression_shap_values, X_test, feature_names=df_plot.columns[1:], class_names=class_names)
```

```python
print(y_test[0:20])
```

```python
sample_number = 5
num_features = len(df_plot.columns[1:])
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=df_plot.columns[1:], class_names=class_names, discretize_continuous=False)
explanation = explainer.explain_instance(X_test[sample_number], clf_trained.predict_proba, num_features=num_features, top_labels=1)
print(f"True label: {list(mapping_dict.keys())[y_test[sample_number]]}")  # Slightly hacky way to get the correct crop type
explanation.show_in_notebook(show_table=True, show_all=True)
```

```python
lda_components = 4
lda_model = LDA(n_components = lda_components)
X_train_lda = lda_model.fit_transform(X_train, y_train)  # Find the transformation parameters from the training data
X_test_lda = lda_model.transform(X_test)  # Transform the test data

# For the plotting parts, we use both training and test data
X_lda = lda_model.transform(X)  # Perform transform on entire dataset
print(f"Explained variation per principal component: {lda_model.explained_variance_ratio_}")
print(f"Shape of training features: {np.shape(X_lda)}")
print(f"Shape of test features: {np.shape(X_lda)}")
df_lda = pd.DataFrame(data = X_lda)
df_lda['LabelID'] = y
for key, value in mapping_dict.items():
    df_lda.loc[df_lda['LabelID'] == value, 'Crop type'] = key
df_lda = df_lda.drop(columns=['LabelID'])
df_lda
```

```python
df_plot_lda_balanced = pd.DataFrame(columns=df_lda.columns)
for crop_type in class_names:
    df_plot_lda_balanced = pd.concat([df_plot_lda_balanced, df_lda[df_lda['Crop type'] == crop_type].sample(100)])
df_plot_lda_balanced['Crop type'].value_counts()
```

```python
g = sns.PairGrid(df_plot_lda_balanced, hue='Crop type')
g.map_diag(sns.kdeplot)
g.map_lower(plt.scatter)
g.map_upper(sns.kdeplot)
g.add_legend()
```

```python
plt.figure(figsize=(5, 5))
sns.heatmap(df_lda.corr(), annot=True)
```

```python
# Instantiate and evaluate classifier
from sklearn.linear_model import LogisticRegression          
clf = LogisticRegression(solver='lbfgs', multi_class='auto', n_jobs=32, max_iter=1000)
clf_trained_lda, _, accuracy_test, results_report, conf_matrix = evaluate_classifier(
    clf, X_train_lda, X_test_lda, y_train, y_test, class_names, feature_scale=False, plot_conf_matrix=True,
    print_classification_report=True)
```

```python
# https://slundberg.github.io/shap/notebooks/linear_explainer/Sentiment%20Analysis%20with%20Logistic%20Regression.html
shap.initjs()
logistic_regression_explainer = shap.LinearExplainer(clf_trained_lda, X_test_lda, feature_perturbation='interventional')
logistic_regression_shap_values = logistic_regression_explainer.shap_values(X_test_lda)
shap.summary_plot(logistic_regression_shap_values, X_test_lda_samples)
```

```python
print(y_test[:20])
```

```python
sample_number = 61 
explainer = lime.lime_tabular.LimeTabularExplainer(X_train_lda, feature_names=range(lda_components), class_names=class_names, discretize_continuous=True)
explanation = explainer.explain_instance(X_test_lda[sample_number], clf_trained_lda.predict_proba, num_features=4, top_labels=5)
print(f"True label: {list(mapping_dict.keys())[y_test[sample_number]]}")  # Slightly hacky way to get the correct crop type
explanation.show_in_notebook(show_table=True, show_all=True)
```

```python

```

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# For some reason, it randomly chosses between regression and classification. Run it multiple times to hit classification.
# dabl.plot(X_train, y_train)
```

```python
# fc = dabl.SimpleClassifier(random_state=RANDOM_SEED)
# fc.fit(X_train, y_train)
```

```python
# dabl.explain(fc, X_test, y_test)
```

```python
X_train_df = pd.DataFrame(X_train)
X_train_df.head(5)
```

```python
X_train_df.describe()
```

```python
X_train_df.corr()
```

```python
with sns.plotting_context(rc={"axes.labelsize":16}):  # Temporarily change the font size for seaborn plots
    # Select the first 8 columns 
    df_cancer_plot_features = X_train_df.iloc[:, :]
    
    # Create pairgrid with hue set to show the two different diagnoses
    g = sns.PairGrid(df_cancer_plot_features, palette="Set2")
    g = g.map_offdiag(plt.scatter, edgecolor="w", s=40)

    # Show histograms of the data on the diagonal and plot the figure
    g = g.map_diag(plt.hist, edgecolor="w")
    plt.show()
```

```python
model = LDA(n_components = 8)
X_lda = model.fit_transform(X_test, y_test)
print('Explained variation per principal component (2D): {}'.format(model.explained_variance_ratio_))
principalDf = pd.DataFrame(data = X_lda)
finalDf = pd.concat([principalDf, df_sklearn_remapped[['Crop type']]], axis = 1)
finalDf = finalDf.loc[finalDf['Crop type'].isin(crops)]
```

```python
principalDf
```

```python
g = sns.PairGrid(principalDf.sample(1000))
g.map_diag(sns.kdeplot)
g.map_lower(sns.kdeplot)
g.map_upper(plt.scatter)
```

```python
g = sns.PairGrid(pd.DataFrame(X_test).sample(1000))
g.map_diag(sns.kdeplot)
g.map_lower(sns.kdeplot)
g.map_upper(plt.scatter)
```

```python

```

```python
# Get classfication report as pandas df
df_results = pd.DataFrame(results_report).transpose()  

# Round the values to 2 decimals
df_results = df_results.round(2).astype({'support': 'int32'})  

# Correct error when creating df (acc. is copied into 'precision')
df_results.loc[df_results.index == 'accuracy', 'precision'] = ''  

# Correct error when creating df (acc. is copied into 'recall')
df_results.loc[df_results.index == 'accuracy', 'recall'] = ''  

# Correct error when creating df (acc. is copied into 'recall')
df_results.loc[df_results.index == 'accuracy', 'recall'] = ''  

# The number of samples ('support') was incorrectly parsed in to dataframe
df_results.loc[df_results.index == 'accuracy', 'support'] = df_results.loc[
    df_results.index == 'macro avg', 'support'].values

# Print df in latex format (I normally add a /midrule above accuracy manually)
print(df_results.to_latex(index=True))  
```

```python
df_results.loc[df_results.index == 'accuracy', 'precision'] = ''
df_results.loc[df_results.index == 'accuracy', 'recall'] = ''
df_results.loc[df_results.index == 'accuracy', 'support'] = df_results.loc[
    df_results.index == 'macro avg', 'support'].values
df_results
```

```python
try:
    from eli5 import show_weights
except:
    !conda install -y eli5
```

```python
import eli5
from eli5.sklearn import PermutationImportance

feature_names = df_sklearn.columns[3:]
perm = PermutationImportance(clf, random_state=1).fit(X_test, y_test)
eli5.explain_weights(clf, feature_names=list(feature_names), target_names=class_names)
# Look at https://eli5.readthedocs.io/en/latest/autodocs/formatters.html#eli5.formatters.html.format_as_html

# IMPORTANT: LOOK HERE TO FIND IMPORTANCE FOR INDIVIDUAL CLASSES:
#            https://stackoverflow.com/questions/59245580/eli5-explain-weights-does-not-returns-feature-importance-for-each-class-with-skl
```

```python
df_explanation = eli5.formatters.as_dataframe.explain_weights_df(perm, feature_names=list(feature_names))
df_explanation = df_explanation.sort_values(by=['feature'])
df_explanation['polarization'] = ''
features = df_explanation['feature'].unique()
for feature in features:
    if feature[-5:] == 'VV-VH':
        df_explanation.loc[df_explanation['feature'] == feature, 'polarization'] = 'VV-VH'
        df_explanation = df_explanation.replace(feature, feature[:-6])
    elif feature[-2:] == 'VV':
        df_explanation.loc[df_explanation['feature'] == feature, 'polarization'] = 'VV'
        df_explanation = df_explanation.replace(feature, feature[:-3])
    else:
        df_explanation.loc[df_explanation['feature'] == feature, 'polarization'] = 'VH'
        df_explanation = df_explanation.replace(feature, feature[:-3])
        
# OLD CODE:
#df_explanation_vh = df_explanation.iloc[::3]
#df_explanation_vh['polarization'] = 'VH'
#df_explanation_vh['feature'] = df_explanation_vh['feature'].map(lambda x: str(x)[:-3])
#df_explanation_vv = df_explanation.iloc[1::3]
#df_explanation_vv['polarization'] = 'VV'
#df_explanation_vv['feature'] = df_explanation_vv['feature'].map(lambda x: str(x)[:-3])
#df_explanation_vvvh = df_explanation.iloc[2::3]
#df_explanation_vvvh['polarization'] = 'VV-VH'
#df_explanation_vvvh['feature'] = df_explanation_vvvh['feature'].map(lambda x: str(x)[:-6])
#df_explanation = pd.concat([df_explanation_vh, df_explanation_vv, df_explanation_vvvh])
```

```python
df_explanation.head(3)
```

```python
plt.figure(figsize=(24, 8))
plt.xticks(rotation=90, horizontalalignment='center')
ax = sns.lineplot(x='feature', y='weight', hue='polarization', data=df_explanation, ci='sd')
```

```python
#df_explanation = eli5.formatters.as_dataframe.explain_prediction_df(perm, feature_names=list(feature_names))
```

```python
# Show the calculated stds in the df as confidence interval on the plot
# https://stackoverflow.com/questions/58399030/make-a-seaborn-lineplot-with-standard-deviation-confidence-interval-specified-f
#lower_bound = [M_new_vec[i] - Sigma_new_vec[i] for i in range(len(M_new_vec))]
#upper_bound = [M_new_vec[i] + Sigma_new_vec[i] for i in range(len(M_new_vec))]
#plt.fill_between(x_axis, lower_bound, upper_bound, alpha=.3)
```

```python
df_explanation = eli5.formatters.as_dataframe.explain_weights_df(clf, feature_names=list(feature_names), target_names=class_names)
df_explanation = df_explanation.sort_values(by=['feature', 'target'])
df_explanation['polarization'] = ''
features = df_explanation['feature'].unique()
features = features[:-1]  # The last features are the bias values
df_bias_values = df_explanation[df_explanation['feature'] == '<BIAS>']

df_explanation = df_explanation[df_explanation['feature'] != '<BIAS>']
for feature in features:
    if feature[-5:] == 'VV-VH':
        df_explanation.loc[df_explanation['feature'] == feature, 'polarization'] = 'VV-VH'
        df_explanation = df_explanation.replace(feature, feature[:-6])
    elif feature[-2:] == 'VV':
        df_explanation.loc[df_explanation['feature'] == feature, 'polarization'] = 'VV'
        df_explanation = df_explanation.replace(feature, feature[:-3])
    else:
        df_explanation.loc[df_explanation['feature'] == feature, 'polarization'] = 'VH'
        df_explanation = df_explanation.replace(feature, feature[:-3])
```

```python
df_bias_values
```

```python
data = df_explanation[df_explanation['polarization'] == 'VH']
#data = data.loc[data['target'].isin(['Forest', 'Maize', 'Rapeseed'])]
plt.figure(figsize=(24, 8))
plt.xticks(rotation=90, horizontalalignment='center')
ax = sns.lineplot(x='feature', y='weight', hue='target', data=data, ci='sd')
```

```python
data = df_explanation[df_explanation['polarization'] == 'VH']
data = data.loc[data['target'].isin(['Barley', 'Wheat', 'Forest', 'Potato'])]
plt.figure(figsize=(24, 8))
plt.xticks(rotation=90, horizontalalignment='center')
ax = sns.lineplot(x='feature', y='weight', hue='target', data=data, ci='sd')
```

```python

```
