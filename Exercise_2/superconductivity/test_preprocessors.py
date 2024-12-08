import numpy as np
import pandas as pd
from utils import load_dataset, setup_logging
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   MaxAbsScaler, PowerTransformer)

# Set up logging
logger = setup_logging("test_preprocessor")

# load dataset
X, y = load_dataset()

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

mae = []
mse = []
rmse = []
for name, scaler in scalers:
    pipeline = Pipeline([
        ("preprocessor", scaler),
        ("rf", RandomForestRegressor(random_state=1234))
    ])
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1234)
    scores = cross_validate(pipeline, X, y, scoring=["neg_mean_absolute_error", "neg_mean_squared_error", "neg_root_mean_squared_error"], cv=cv, n_jobs=-1)
    mae.append((name, -np.mean(scores['test_neg_mean_absolute_error']), np.std(scores['test_neg_mean_absolute_error'])))
    mse.append((name, -np.mean(scores['test_neg_mean_squared_error']), np.std(scores['test_neg_mean_squared_error'])))
    rmse.append((name, -np.mean(scores['test_neg_root_mean_squared_error']), np.std(scores['test_neg_root_mean_squared_error'])))
logger.info(f"\nAccuracy:")
# Log the results
logger.info(f"\nMean Absolute Error (MAE):")
logger.info(pd.DataFrame(mae, columns=['scaler', 'mean', 'std']).sort_values(by='mean', ascending=True))

logger.info(f"\nMean Squared Error (MSE):")
logger.info(pd.DataFrame(mse, columns=['scaler', 'mean', 'std']).sort_values(by='mean', ascending=True))

logger.info(f"\nRoot Mean Squared Error (RMSE):")
logger.info(pd.DataFrame(rmse, columns=['scaler', 'mean', 'std']).sort_values(by='mean', ascending=True))