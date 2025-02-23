import pandas as pd
import numpy as np
from tpot import TPOTRegressor
from sklearn.model_selection import RepeatedKFold
from utils import load_dataset, setup_logging
from sklearn.metrics import root_mean_squared_error, make_scorer


logger = setup_logging("comp_tpot")

X, y = load_dataset()

tpot_config = {'sklearn.neighbors.KNeighborsRegressor': {'n_neighbors': [3,5,10,20], 'p': [1,2,3,4,5]},
               'sklearn.svm.LinearSVR': {'max_iter': [1000,2000,5000,10000], 'C': [10**i for i in range(-3, 4)], 'fit_intercept': [True, False]},
               'sklearn.ensemble.RandomForestRegressor': {'n_estimators': [10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,500,600,700,800,900],
                                                          'max_depth': [5,10,20,30,50,75,100,150,200],
                                                          'max_features': ['sqrt', 'log2', None]},
                'sklearn.linear_model.Ridge': {'alpha': [10**i for i in range(-3, 4)], 'fit_intercept': [True, False]},
                'sklearn.linear_model.Lasso': {'alpha': [10**i for i in range(-3, 4)], 'fit_intercept': [True, False]}}

custom_score = make_scorer(root_mean_squared_error, greater_is_better=False)

rkf = RepeatedKFold(n_splits=4, n_repeats=1, random_state=42)


tpot = TPOTRegressor(generations=20, cv=rkf, population_size=100, verbosity=2, random_state=42, max_time_mins=150, config_dict=tpot_config, scoring=custom_score, template='Regressor', n_jobs=-1)
tpot.fit(X, y)


pd.set_option('display.max_colwidth', None)
model_dict = list(tpot.evaluated_individuals_.items())
results = pd.DataFrame()

for model in model_dict:
    model_name = model[0]
    model_info = model[1]
    cv_score = model[1].get('internal_cv_score')  # Pull out cv_score as a column (i.e., sortable)
    results = results._append({'model': model_name,
                                        'cv_score': cv_score,
                                        'model_info': model_info,},
                                       ignore_index=True)

results = results.sort_values('cv_score', ascending=False)

logger.info(f"History:\n{results}\n\n")

