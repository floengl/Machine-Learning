from utils import load_training_dataset, setup_logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
import pandas as pd

# Set up logging
logger = setup_logging("tune_random_forest")

# load dataset
X, y = load_training_dataset()

# define estimator
estimator = Pipeline([
    ("preprocessor", MinMaxScaler()),
    ("model", RandomForestClassifier(random_state=1234))
])

# search space
search_space = {
    "model__n_estimators": Integer(100, 3000),
    "model__max_depth": Categorical([None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
    "model__min_samples_split": Integer(2, 20),
    "model__min_samples_leaf": Integer(1, 5),
    "model__max_features": Categorical([None, "sqrt", "log2"]),
}

# cross validation strategy
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1234)

for score in ["accuracy", "recall"]:
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
    df = pd.DataFrame(optimizer.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    logger.info("CV results:")
    logger.info(df.sort_values("rank_test_score").to_string())
    logger.info("\n")
