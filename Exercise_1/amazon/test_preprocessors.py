from utils import load_training_dataset, setup_logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
# Set up logging
logger = setup_logging("test_preprocessor")

# load dataset
X, y = load_training_dataset()

#disable warnigns for unseen data in test dataset
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn.preprocessing._encoders')
# Suppress the specific ConvergenceWarning from sklearn.svm._base
warnings.filterwarnings(action='ignore', category=ConvergenceWarning, module='sklearn.svm._base')

# max abs scaler
max_abs_scaler = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("max_abs_scaler", MaxAbsScaler())
])

# standard scaler
standard_scaler = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("standard_scaler", StandardScaler(with_mean=False))
])

# no scaling
no_scaling = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough"))
])

# robust scaler
robust_scaler = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("robust_scaler", RobustScaler())
])
# power transformer
power_transformer = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("power_transformer", PowerTransformer())
])

# quantile transformer
quantile_transformer = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("quantile_transformer", QuantileTransformer())
])
minmax_scaler = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("minmax_scaler", MinMaxScaler())
])

# PCA
pca = Pipeline([
    ("pca", PCA(n_components=0.95))
])

# no dimensionality reduction
no_dim_reduction = Pipeline([
    ("no_dim_reduction", "passthrough")
])

# scalers to test
scalers = []
scalers.append(("max_abs_scaler", max_abs_scaler))
scalers.append(("standard_scaler", standard_scaler))
scalers.append(("none", no_scaling))
scalers.append(("robust_scaler", robust_scaler))
scalers.append(("power_transformer", power_transformer))    
scalers.append(("quantile_transformer", quantile_transformer))  

# dimensionality reduction to test
dim_reductions = []
dim_reductions.append(("pca", pca))
dim_reductions.append(("none", no_dim_reduction))

# models to test with
models = [
    ("RandomForest", RandomForestClassifier(random_state=1234)),
    ("LinearSVC", LinearSVC(random_state=1234)),
    ("Ridge", RidgeClassifier(random_state=1234))
]


for model_name, model in models:
    accuracy = []
    f1 = []
    for name_scaler, scaler in scalers:
        for name_dim_reduction, dim_reduction in dim_reductions:
            print(name_dim_reduction)
            print(name_scaler)
            print(model_name)
            pipeline = Pipeline([
                ("scaler", scaler),
                ("dim_reduction", dim_reduction),
                ("model", model)
            ])
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1234)
            scores = cross_validate(pipeline, X, y, scoring=["accuracy", "f1_macro"], cv=cv, n_jobs=-1)
            accuracy.append(
                (name_scaler, name_dim_reduction, np.mean(scores['test_accuracy']), np.std(scores['test_accuracy']))
            )
            f1.append(
                (name_scaler, name_dim_reduction, np.mean(scores['test_f1_macro']), np.std(scores['test_f1_macro']))
            )

    logger.info(f"\nAccuracy {model_name}:")
    logger.info(pd.DataFrame(accuracy, columns=['scaler', 'dim_reduction', 'mean', 'std'])
                  .sort_values(by='mean', ascending=False))
    logger.info(f"\nF1 {model_name}:")
    logger.info(pd.DataFrame(f1, columns=['scaler', 'dim_reduction', 'mean', 'std'])
                  .sort_values(by='mean', ascending=False))
