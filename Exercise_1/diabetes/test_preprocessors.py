import numpy as np
import pandas as pd
from utils import load_training_dataset, setup_logging
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   MaxAbsScaler, PowerTransformer)

# Set up logging
logger = setup_logging("test_preprocessor")

# load dataset
X, y = load_training_dataset()

# scalers to test
scalers = [
    ("StandardScaler", StandardScaler()),
    ("MinMaxScaler", MinMaxScaler()),
    ("MaxAbsScaler", MaxAbsScaler()),
    ("RobustScaler", RobustScaler()),
    ("PowerTransformer", PowerTransformer()),
    ("QuantileTransformer", PowerTransformer()),
    ("None", Pipeline([("none", "passthrough")]))
]

# models to test with
models = [
    ("rf", RandomForestClassifier(random_state=1234)),
    ("LinearSVC", LinearSVC(random_state=1234, max_iter=10000, dual=False)),
    ("Ridge", RidgeClassifier(random_state=1234))
]

for model_name, model in models:
    accuracy = []
    recall = []
    for name, scaler in scalers:
        pipeline = Pipeline([
            ("preprocessor", scaler),
            ("model", model)
        ])
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1234)
        scores = cross_validate(pipeline, X, y, scoring=["accuracy", "recall"], cv=cv, n_jobs=-1)
        accuracy.append((name, np.mean(scores['test_accuracy']), np.std(scores['test_accuracy'])))
        recall.append((name, np.mean(scores['test_recall']), np.std(scores['test_recall'])))
    logger.info(f"\nAccuracy {model_name}:")
    logger.info(pd.DataFrame(accuracy, columns=['scaler', 'mean', 'std'])
                  .sort_values(by='mean', ascending=False))
    logger.info(f"\nRecall {model_name}:")
    logger.info(pd.DataFrame(recall, columns=['scaler', 'mean', 'std'])
                  .sort_values(by='mean', ascending=False))
