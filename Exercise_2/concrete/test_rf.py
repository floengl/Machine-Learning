from utils import load_dataset, setup_logging
from utils import logger
from random_forest import ourRandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

if __name__ == "__main__":

    logger = setup_logging("test_random_forest")
    X, y = load_dataset()
   # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

    scalers = [
        ("RobustScaler", RobustScaler()),
        ("None", Pipeline([("none", "passthrough")]))
    ]

    nb_trees=40
    boot_type= True
    nb_samples="Full"
    max_workers=12

    logger.info(f"\nnb_trees={nb_trees}, nb_samples={nb_samples}, boot_type={boot_type}, max_workers={max_workers}")
    # models to test with
    models = [
        ("ourRFR", ourRandomForestRegressor(nb_trees=nb_trees, nb_samples=nb_samples, boot_type=boot_type, max_workers=max_workers, random_state=1234)),
        ("scikit_RFR", RandomForestRegressor(random_state=1234)),
        ("scikit_KNN", KNeighborsRegressor())
    ]

    for model_name, model in models:
        mse = []
        rmse = []
        for name, scaler in scalers:

            pipeline = Pipeline([
                ("preprocessor", scaler),
                ("model", model)
            ])

            cv=RepeatedKFold(n_splits=4, n_repeats=3, random_state=1234)

            # Perform cross-validation
            cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=[ "neg_mean_squared_error"], return_train_score=True, n_jobs=-1)
            
            mse.append((name, -np.mean(cv_results['test_neg_mean_squared_error']), np.std(cv_results['test_neg_mean_squared_error'])))
            root_mean_squared_error = np.sqrt(-np.mean(cv_results['test_neg_mean_squared_error']))
            rmse.append((name, root_mean_squared_error))

        logger.info(f"\nMSE {model_name}:")
        logger.info(pd.DataFrame(mse, columns=['preprocessor', 'mean', 'std'])
                    .sort_values(by='mean', ascending=True))
        logger.info(f"\nRMSE {model_name}:")
        logger.info(pd.DataFrame(rmse, columns=['preprocessor', 'rmean'])
                    .sort_values(by='rmean', ascending=True))
        



   