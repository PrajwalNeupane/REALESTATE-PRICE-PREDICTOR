import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from joblib import dump, load

## Creating a pandas dataframe and removing NaN Values
housing = pd.read_csv("data.csv")
df = housing
df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
housing = df

## Train-Test Splitting
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

## Using only train as main dataframe aka Housing
housing = strat_train_set.copy()

## Separating label and data to fit
housing = strat_train_set.drop('MEDV', axis=1)
housing_labels = strat_train_set["MEDV"].copy()

## Pipeline init

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

## Using Pipeline 
housing_num_tr = my_pipeline.fit_transform(housing)

## Implementing Models

models = [DecisionTreeRegressor, LinearRegression, RandomForestRegressor]
f = open("ModelsOutcome.txt", "a")
counter = 1
means = {}

for model in models:
    curr_model = model()
    curr_model.fit(housing_num_tr, housing_labels)

    ## Evaluating models
    scores = cross_val_score(curr_model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-scores)

    ## Storing result in dict
    means[str(curr_model)] = rmse_scores.mean()

    ## Writing To File 
    f.write(f"\n{counter}. {str(curr_model)}\n")
    f.write(f"Mean : {str(rmse_scores.mean())}\n")
    f.write(f"Standard Deviation : {str(rmse_scores.std())}\n")

    counter += 1

## Choosing the Model and storing it in variable 'model'
means_values = list(means.values())
means_keys = list(means.keys())

optimal_model_rmse_index = means_values.index(min(means_values))
optimal_model = means_keys[optimal_model_rmse_index]

f.write(f"\nOptimal Model: {str(optimal_model)}\n")
f.write(f"Mean: {min(means_values)}")

for i in range(len(models)):
    if str(models[i] == optimal_model):
        model = models[i]

## dumping chosen model
dump(model, 'Dragon.joblib')

f.close()
