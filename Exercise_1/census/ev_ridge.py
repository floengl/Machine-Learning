from utils import load_training_dataset, setup_logging, categorical, numeric
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.linear_model import RidgeClassifier
from skopt import BayesSearchCV
from skopt.space import Real
import pandas as pd
import numpy as np

# Set up logging
logger = setup_logging("ev_ridge")

# load dataset
X, y = load_training_dataset()

# categorical preprocessing: nan is its own category
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="nan")),
    ("one_hot_encoder", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))
])

# numeric preprocessing: standard scaler
numeric_transformer = Pipeline([
    ("StandardScaler", QuantileTransformer())
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
    ("model", RidgeClassifier(random_state=1234, alpha=11.333, fit_intercept=True, class_weight="balanced"))
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