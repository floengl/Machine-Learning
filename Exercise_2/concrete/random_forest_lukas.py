#!/usr/bin/env python3

import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from utils import load_dataset, setup_logging
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from utils import logger



class RandomForestRegressor(object):
    """
    :param  nb_trees:       Number of decision trees to use
    :param  nb_samples:     Number of samples to give to each tree
    :param  max_depth:      Maximum depth of the trees
    :param  max_workers:    Maximum number of processes to use for training
    """
    def __init__(self, nb_trees, nb_samples, max_depth=-1, max_workers=1):
        self.trees = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.max_depth = max_depth
        self.max_workers = max_workers

    """
    Trains self.nb_trees number of decision trees.
    :param  X:   Features
    :param  y:   Target values
    """
    def fit(self, X, y):
        data = list(zip(X.values, y.values))
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            rand_fts = map(lambda x: [x, random.sample(data, min(self.nb_samples, len(data)))],
                           range(self.nb_trees))
            self.trees = list(executor.map(self.train_tree, rand_fts))

    """
    Trains a single tree and returns it.
    :param  data:   A List containing the index of the tree being trained
                    and the data to train it
    """
    def train_tree(self, data):
        if self.max_depth == -1:
            tree = DecisionTreeRegressor()
        else:
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
        X = np.array([d[0] for d in data[1]])  # Features
        y = np.array([d[1] for d in data[1]])  # Target
        tree.fit(X, y)
        return tree

    """
    Returns a prediction for the given feature. The result is the average of
    the predictions from all trees.
    :param  feature:    The features used to predict
    """
    def predict(self, feature):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict([feature])[0])
        return np.mean(predictions)


def test_rf():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    rf = RandomForestRegressor(nb_trees=60, nb_samples=20, max_workers=12)
    rf.fit(X_train, y_train)

    predictions = [rf.predict(feature) for feature in X_test.values]
    meansquarederror = np.mean((predictions - y_test.values) ** 2)
    rootmeansquarederror = np.sqrt(meansquarederror)
    absoluteerror = np.mean(np.abs(predictions - y_test.values))
    logger.info(f"Mean Squared Error: {meansquarederror}")
    logger.info(f"Root Mean Squared Error: {rootmeansquarederror}")
    logger.info(f"Absolute Error: {absoluteerror}")


# Run the test function
if __name__ == "__main__":
    logger = setup_logging("random_forest")
    test_rf()


