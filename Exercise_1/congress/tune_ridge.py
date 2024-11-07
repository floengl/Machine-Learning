from utils import load_training_dataset, setup_logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import RidgeClassifier
from skopt import BayesSearchCV
from skopt.space import Real
import pandas as pd

# Set up logging
logger = setup_logging("tune_ridge")

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
    ("model", RidgeClassifier(random_state=1234))
])

# search space
search_space = {
    "model__alpha": Real(0, 100),
    "model__fit_intercept": [True, False]
}

# cross validation strategy
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1234)

for score in ["accuracy", "f1"]:
    # optimizer
    optimizer = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_space,
        scoring=score,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=1234
    )

    # fit
    optimizer.fit(X, y)

    # log best score and parameters
    logger.info(f"Scoring: {score}")
    logger.info(f"Best score: {optimizer.best_score_}")
    logger.info(f"best params: {optimizer.best_params_}")
    df = pd.DataFrame(optimizer.cv_results_)[["param_model__alpha", "param_model__fit_intercept",
                                              "mean_test_score", "std_test_score", "rank_test_score"]]
    logger.info("CV results:")
    logger.info(df.sort_values("rank_test_score").to_string())
    logger.info("\n")
