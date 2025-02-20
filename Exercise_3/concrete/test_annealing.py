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
from sklearn.metrics import f1_score
from utils import load_dataset, setup_logging
from utils import logger

from simulated_annealing import *

logger = setup_logging("test_annealing")

models = {"rf": RandomForestRegressor,
          "LinearSVR": LinearSVR,
          "Ridge": Ridge,
          "BinaryTree": DecisionTreeRegressor,
          "KNN": KNeighborsRegressor}


start_params = {"rf": {"n_estimators": 100, "max_depth": 10},
                "LinearSVR": {"max_iter": 1000, "C": 1.0},
                "Ridge": {"alpha": 1.0},
                "BinaryTree": {"max_depth": 5, "min_samples_split": 2},
                "KNN": {"n_neighbors": 5}}

params_vals = {"rf": {"n_estimators": [10,20,50,100,200], "max_depth": [5,10,20,50]},
                 "LinearSVR": {"max_iter": [1000,2000,5000], "C": [0.1,1.0,10.0]},
                "Ridge": {"alpha": [0.1,1.0,10.0]},
                "BinaryTree": {"max_depth": [5,10,20,50], "min_samples_split": [2,5,10]},
                "KNN": {"n_neighbors": [3,5,10,20]}}

X, y = load_dataset()
Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(X, y, test_size=0.2, random_state=42)

model, results = simulate_annealing(start_params, params_vals, Xtrain, Xvalid, Ytrain, Yvalid, train_model, models =  models, maxiters=1000, T_0=400)

logger.info(f"Best model: {model}")
logger.info(f"Best score: {results['Best Metric'].min()}")

# Convert the DataFrame to a string with all rows and columns displayed
results_str = results.to_string()

# Log the entire DataFrame
logger.info(f"History:\n{results_str}")

print(model)
print(results)




