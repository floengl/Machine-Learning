#!/usr/bin/env python3

import random
from concurrent.futures import ProcessPoolExecutor
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
from utils import load_dataset, setup_logging, load_dataset_notopcolumn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from utils import logger
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold
from regression_tree import RegressionTree
from sklearn.ensemble import RandomForestRegressor



class ourRandomForestRegressor(object):
    """
    :param  nb_trees:       Number of decision trees to use
    :param  nb_samples:     Number of samples to give to each tree
    :param  max_depth:      Maximum depth of the trees
    :param  max_workers:    Maximum number of processes to use for training
    """
    def __init__(self, nb_trees, nb_samples, max_depth=-1, max_workers=1, random_state=None):
        self.trees = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.max_depth = max_depth
        self.max_workers = max_workers
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        #if random_state!=None:
        #    random.seed(random_state)
    """
    Trains self.nb_trees number of decision trees.
    :param  X:   Features
    :param  y:   Target values
    """
    def fit(self, X, y):
        if type(X) == np.ndarray:
            data = list(zip(X, y))
        else:
            X= X.values
            y = y.values
            data = list(zip(X, y))
        

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            rand_fts = [self.rng.choice(data, size=min(self.nb_samples, len(data)), replace=False) for _ in range(self.nb_trees)]
            random_states = self.rng.integers(low=0, high=1e6, size=self.nb_trees)
            self.trees = list(executor.map(self.train_tree, rand_fts, random_states))



    """
    Trains a single tree and returns it.
    :param  data:   A List containing the index of the tree being trained
                    and the data to train it
    """
    def train_tree(self, data, random_state):
        if self.max_depth == -1:
            tree = RegressionTree()
        else:
            tree = RegressionTree(max_depth=self.max_depth, random_state=random_state)
            
        X,y = zip(*data)
        X = np.array(X)
        y = np.array(y)
        tree.fit(X, y)
        return tree

    """
    Returns a prediction for the given feature. The result is the average of
    the predictions from all trees.
    :param  feature:    The features used to predict
    """
    def predict(self, feature):
        if isinstance(feature, pd.DataFrame):
           feature = feature.values
           print(feature.shape)
        predictions = np.array([tree.predict(feature) for tree in self.trees])
        return np.mean(predictions, axis=0)




logger = setup_logging("random_forest")
X, y = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

scalers = [
    ("RobustScaler", RobustScaler()),
    ("None", Pipeline([("none", "passthrough")]))
]

mae = []
mse = []
rmse = []

for name, scaler in scalers:
    pipeline = Pipeline([
        ("preprocessor", scaler),
        ("rf", ourRandomForestRegressor(nb_trees=40, nb_samples=1000, max_workers=12, random_state=1234))
    ])
    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)



    # Predict on the test data
    predictions = pipeline.predict(X_test)
        
    mean_absolute_error = np.mean(np.abs(predictions - y_test))
    mean_squared_error = np.mean((predictions - y_test) ** 2)
    root_mean_squared_error = np.sqrt(mean_squared_error)



    mae.append((name, mean_absolute_error))
    mse.append((name, mean_squared_error))
    rmse.append((name, root_mean_squared_error))



tree = DecisionTreeRegressor(random_state=1234)
tree.fit(X_train, y_train)
tree_predictions = tree.predict(X_test)    
rf = RandomForestRegressor(n_estimators=40,random_state=1234)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)



brmae = np.mean(np.abs(tree_predictions - y_test))
brmse= np.mean((tree_predictions - y_test) ** 2)
brrmse = np.sqrt(brmse)

rfmae = np.mean(np.abs(rf_predictions - y_test))
rfmse= np.mean((rf_predictions - y_test) ** 2)
rfrmse = np.sqrt(rfmse)


logger.info("\nscikitlearn:")
logger.info("\nMean Absolute Error (MAE):")
logger.info(pd.DataFrame([["binarytree", brmae], ["random_forest", rfmae]], columns=['model', 'mean']))

logger.info("\nMean Squared Error (MSE):")
logger.info(pd.DataFrame([["binarytree", brmse], ["random_forest", rfmse]], columns=['model', 'mean']))

logger.info("\nRoot Mean Squared Error (RMSE):")
logger.info(pd.DataFrame([["binarytree", brrmse], ["random_forest", rfrmse]], columns=['model', 'mean']))

# Log the results
logger.info(f"\nMean Absolute Error (MAE):")
logger.info(pd.DataFrame(mae, columns=['scaler', 'mean']).sort_values(by='mean', ascending=True))

logger.info(f"\nMean Squared Error (MSE):")
logger.info(pd.DataFrame(mse, columns=['scaler', 'mean']).sort_values(by='mean', ascending=True))

logger.info(f"\nRoot Mean Squared Error (RMSE):")
logger.info(pd.DataFrame(rmse, columns=['scaler', 'mean']).sort_values(by='mean', ascending=True))




