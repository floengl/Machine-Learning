from utils import load_training_dataset, setup_logging, categorical, numeric
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

# categorical preprocessing: nan is its own category
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="nan")),
    ("one_hot_encoder", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))
])

# numeric preprocessing: standard scaler
numeric_transformer = Pipeline([
    ("StandardScaler", StandardScaler())
])

# define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric),
        ("categorical", categorical_transformer, categorical)
    ]
)

# define estimator
estimator = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearSVC(random_state=1234, max_iter=100000, C=679.520, class_weight="balanced", fit_intercept=True, dual=False))
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
    