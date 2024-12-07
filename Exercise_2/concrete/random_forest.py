#!/usr/bin/env python3

from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from regression_tree import RegressionTree



class ourRandomForestRegressor(object):
    """
    :param  nb_trees:       Number of decision trees to use
    :param  nb_samples:     Number of samples to give to each tree
    :param  max_depth:      Maximum depth of the trees
    :param  max_workers:    Maximum number of processes to use for training
    """
    def __init__(self, nb_trees,  nb_samples = "Full", max_depth=-1, max_workers=1, random_state=None, boot_type = True, min_samples_split=2, max_features=None):
        self.trees = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_workers = max_workers
        self.random_state = random_state
        self.boot_type = boot_type
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
        
        length = len(data)

      
        if isinstance(self.nb_samples, float) and 0<self.nb_samples<1:
            nb_samples = int(length*self.nb_samples)
        elif isinstance(self.nb_samples, int):
            nb_samples = min(self.nb_samples, length)
        else: nb_samples = length

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            indices = np.arange(length)
            #chooses random indices to bootstrap the data
            rand_ind = [self.rng.choice(indices, size=nb_samples, replace=self.boot_type) for _ in range(self.nb_trees)]
            #extracts the data for the chosen indices
            bootstrap_data = [[data[i] for i in ind] for ind in rand_ind]
            #initializes randomly generated states for each tree   
            random_states = self.rng.integers(low=0, high=1e6, size=self.nb_trees)
            #builds the trees
            self.trees = list(executor.map(self.train_tree, bootstrap_data, random_states))



    """
    Trains a single tree and returns it.
    :param  data:   A List containing the index of the tree being trained
                    and the data to train it
    """
    def train_tree(self, data, random_state):
        if self.max_depth == -1:
            tree = RegressionTree(random_state=random_state, min_samples_split=self.min_samples_split, max_features=self.max_features)
        else:
            tree = RegressionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features , random_state=random_state)
            
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





