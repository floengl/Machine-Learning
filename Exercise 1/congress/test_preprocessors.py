from utils import load_training_dataset, setup_logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
import pandas as pd

# Set up logging
logger = setup_logging("test_preprocessor")

# load dataset
X, y = load_training_dataset()

# nan is category
nan_is_category = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("imputer", SimpleImputer(strategy="constant", fill_value="nan")),
    ("one_hot_encoder", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))
])

# most frequent imputation
most_frequent = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("one_hot_encoder", OneHotEncoder(drop="if_binary"))
])

# preprocessors to test
preprocessors = []
preprocessors.append(("nan_is_category", nan_is_category))
preprocessors.append(("most_frequent", most_frequent))

# models to test with
models = [
    ("rf", RandomForestClassifier(random_state=1234)),
    ("LinearSVC", LinearSVC(random_state=1234)),
    ("Ridge", RidgeClassifier(random_state=1234))
]

for model_name, model in models:
    accuracy = []
    f1 = []
    for name, preprocessor in preprocessors:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1234)
        scores = cross_validate(pipeline, X, y, scoring=["accuracy", "f1"], cv=cv, n_jobs=-1)
        accuracy.append((name, np.mean(scores['test_accuracy']), np.std(scores['test_accuracy'])))
        f1.append((name, np.mean(scores['test_f1']), np.std(scores['test_f1'])))
    logger.info(f"\nAccuracy {model_name}:")
    logger.info(pd.DataFrame(accuracy, columns=['preprocessor', 'mean', 'std'])
                  .sort_values(by='mean', ascending=False))
    logger.info(f"\nF1 {model_name}:")
    logger.info(pd.DataFrame(f1, columns=['preprocessor', 'mean', 'std'])
                  .sort_values(by='mean', ascending=False))
