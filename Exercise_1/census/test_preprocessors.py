import numpy as np
import pandas as pd
from utils import load_training_dataset, setup_logging, categorical, numeric
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.preprocessing import (OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler,
                                   MaxAbsScaler, QuantileTransformer, PowerTransformer)
import warnings
from sklearn.exceptions import ConvergenceWarning

# Set up logging
logger = setup_logging("test_preprocessor")

# load dataset
X, y = load_training_dataset()


# numeric features don't have missing values, so we only need to scale them
numeric_transformers = [
    ("StandardScaler", StandardScaler()),
    ("MinMaxScaler", MinMaxScaler()),
    ("MaxAbsScaler", MaxAbsScaler()),
    ("RobustScaler", RobustScaler()),
    ("PowerTransformer", PowerTransformer()),
    ("QuantileTransformer", QuantileTransformer()),
    ("None", Pipeline([("none", "passthrough")]))
]

# categorical preprocessing: nan is its own category
nan_is_category = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="nan")),
    ("one_hot_encoder", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))
])

# categorical preprocessing: most frequent imputation
most_frequent = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("one_hot_encoder", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))
])

categorical_transformers = [
    ("nan_is_category", nan_is_category),
    ("most_frequent", most_frequent),
]

# models to test with
models = [
    ("RandomForest", RandomForestClassifier(random_state=1234)),
    ("Ridge", RidgeClassifier(random_state=1234)),
    ("LinearSVC", LinearSVC(random_state=1234, max_iter=10000, dual=False))  # dual=False for speed up
]

for model_name, model in models:
    accuracy = []
    f1 = []
    for numeric_transformer_name, numeric_transformer in numeric_transformers:
        for categorical_transformer_name, categorical_transformer in categorical_transformers:
            print(f"Testing {model_name} with {numeric_transformer_name} and {categorical_transformer_name}")
            # create preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ("numeric", numeric_transformer, numeric),
                    ("categorical", categorical_transformer, categorical)
                ]
            )
            # create pipeline
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])
            n_repeats = 1 if model_name == "RandomForest" else 5  # Random Forest is very slow
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=1234)
            scores = cross_validate(pipeline, X, y, scoring=["accuracy", "f1"], cv=cv, n_jobs=-1)
            accuracy.append(
                (numeric_transformer_name, categorical_transformer_name,
                 np.mean(scores['test_accuracy']), np.std(scores['test_accuracy']))
            )
            f1.append(
                (numeric_transformer_name, categorical_transformer_name,
                 np.mean(scores['test_f1']), np.std(scores['test_f1']))
            )
    logger.info(f"\nAccuracy {model_name}:")
    logger.info(pd.DataFrame(accuracy, columns=['numeric_transformer', 'categorical_transformer', 'mean', 'std'])
                .sort_values(by='mean', ascending=False))
    logger.info(f"\nF1 {model_name}:")
    logger.info(pd.DataFrame(f1, columns=['numeric_transformer', 'categorical_transformer', 'mean', 'std'])
                .sort_values(by='mean', ascending=False))
