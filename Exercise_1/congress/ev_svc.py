from utils import load_training_dataset, setup_logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler,
                                   MaxAbsScaler, QuantileTransformer, PowerTransformer)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Set up logging
logger = setup_logging("ev_svc")

# load dataset
X, y = load_training_dataset()

# define preprocessing pipeline
preprocessor = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("one_hot_encoder", OneHotEncoder(drop="if_binary"))
])

# define estimator
estimator = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearSVC(random_state=1234, max_iter=100000, C=1.2, class_weight="balanced", fit_intercept=True, dual=False))
])


accuracy = []
f1 = []
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1234)
scores = cross_validate(estimator, X, y, scoring=["accuracy", "f1"], cv=cv, n_jobs=-1)
accuracy.append((np.mean(scores['test_accuracy']), np.std(scores['test_accuracy'])))
f1.append((np.mean(scores['test_f1']), np.std(scores['test_f1'])))

logger.info(f"\nAccuracy svc:")
logger.info(pd.DataFrame(accuracy, columns=['mean', 'std']).sort_values(by='mean', ascending=False))
logger.info(f"\nF1 svc:")
logger.info(pd.DataFrame(f1, columns=['mean', 'std']).sort_values(by='mean', ascending=False))
    