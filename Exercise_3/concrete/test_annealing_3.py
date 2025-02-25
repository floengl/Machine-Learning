import pandas as pd
import numpy as np
from collections import OrderedDict
from random import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from utils import load_dataset, setup_logging
from utils import logger

from simulated_annealing_3 import *

logger = setup_logging("test_annealing_3")

random_seed = 42

models = {"rf": RandomForestRegressor,
          "LinearSVR": LinearSVR,
          "Ridge": Ridge,
          "Lasso": Lasso,
          "KNN": KNeighborsRegressor}


start_params = {"rf": {"n_estimators": 100, "max_depth": 10, "max_features": "log2"},
                "LinearSVR": {"max_iter": 1000, "C": 1.0, "fit_intercept": True},
                "Ridge": {"alpha": 1.0, "fit_intercept": True},
                "Lasso": {"alpha": 1.0, "fit_intercept": True},
                "KNN": {"n_neighbors": 5, "p": 5}}

params_vals = {"rf": {"n_estimators": [10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,500,600,700,800,900], "max_depth": [5,10,20,30,50,75,100,150,200],
                      "max_features": ["auto", "sqrt", "log2", None]},
                "LinearSVR": {"max_iter": [1000,2000,5000,10000], "C": [10**i for i in range(-3, 4)], "fit_intercept": [True, False]},
                "Ridge": {"alpha": [10**i for i in range(-3, 4)], "fit_intercept": [True, False]},
                "Lasso": {"alpha": [10**i for i in range(-3, 4)], "fit_intercept": [True, False]},
                "KNN": {"n_neighbors": [3,5,10,20], "p": [1,2,3,4,5]}}

X, Y = load_dataset()


model, results, nr_reheats = simulate_annealing(start_params, params_vals, X, Y, models =  models, train_model=train_model_2, maxiters=1000, T_0=400, f=5, n_repeats=1, random_seed=random_seed)

logger.info(f"Random Seed: {random_seed}")
logger.info(f"Best model: {model}")
logger.info(f"Best score: {results['Best Metric'].min()}")
logger.info(f"Number of reheats: {nr_reheats-1}")

# Convert the DataFrame to a string with all rows and columns displayed
results_str = results.to_string()

# Log the entire DataFrame
logger.info(f"History:\n{results_str}")

print(model)
print(results)




