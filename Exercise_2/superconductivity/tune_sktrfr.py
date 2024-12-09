from utils import load_dataset, setup_logging, rse_scorer
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

def main():
    logger = setup_logging("tune_skt_random_forest")
    X, y = load_dataset()

    # define estimator
    estimator = Pipeline([
        ("preprocessor", MaxAbsScaler()),
        ("model", RandomForestRegressor(random_state=1234))
    ])

    # search space
    search_space = {
        "model__n_estimators": Categorical([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]),
        "model__max_depth": Categorical([None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
        "model__min_samples_split": Integer(2, 20),
        "model__max_features": Categorical([None, "sqrt", "log2"]),
        "model__bootstrap": Categorical([True, False]),
    }

    cv=RepeatedKFold(n_splits=4, n_repeats=3, random_state=1234)

    # optimizer
    optimizer = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_space,
        scoring=rse_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=1234,
        n_iter=50
    )

    # fit
    optimizer.fit(X, y)

    # log best score and parameters
    logger.info(f"Scoring: RSE")
    logger.info(f"Best score: {optimizer.best_score_}")
    logger.info(f"best params: {optimizer.best_params_}")
    df = pd.DataFrame(optimizer.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    logger.info("CV results:")
    logger.info(df.sort_values("rank_test_score").to_string())
    logger.info("\n")

    # optimizer
    optimizer = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_space,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=1234,
        n_iter=50
    )

    # fit
    optimizer.fit(X, y)

    # log best score and parameters
    logger.info(f"Scoring: MSE")
    logger.info(f"Best score: {optimizer.best_score_}")
    logger.info(f"best params: {optimizer.best_params_}")
    df = pd.DataFrame(optimizer.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    logger.info("CV results:")
    logger.info(df.sort_values("rank_test_score").to_string())
    logger.info("\n")

if __name__ == "__main__":
    main()