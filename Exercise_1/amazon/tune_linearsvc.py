from utils import load_training_dataset, setup_logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import LinearSVC
from skopt import BayesSearchCV
from skopt.space import Real
import pandas as pd

# Set up logging
logger = setup_logging("new_tune_linearsvc")

# load dataset
X, y = load_training_dataset()

# define preprocessing pipeline
preprocessor = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("powertransformer", PowerTransformer())
])

# define estimator
estimator = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearSVC(random_state=1234, max_iter=100000))
])

# search space
search_space = {
    "model__C": Real(1e-6, 1e+6, 'log-uniform'),
    "model__class_weight": [None, "balanced"],
    "model__fit_intercept": [True, False]#,
    #"model__dual": [True, False]
}

# cross validation strategy
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1234, n_jobs=-1)

for score in ["accuracy", "f1_macro"]:
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
    df = pd.DataFrame(optimizer.cv_results_)[["param_model__C", "param_model__class_weight",
                                              "param_model__fit_intercept", #"param_model__dual", 
                                              "mean_test_score",
                                              "std_test_score", "rank_test_score"]]
    logger.info("CV results:")
    logger.info(df.sort_values("rank_test_score").to_string())
    logger.info("\n")
