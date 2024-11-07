from utils import load_training_dataset, setup_logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
import numpy as np

# Set up logging
logger = setup_logging("test_fit_intercept")

# load dataset
X, y = load_training_dataset()

# ridge classifier
ridge = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("max_abs_scaler", MaxAbsScaler()),
    ("pca", PCA(n_components=0.95)),
    ("model", RidgeClassifier(random_state=1234))
])

# linear SVC
linear_svc = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("max_abs_scaler", MaxAbsScaler()),
    ("model", LinearSVC(random_state=1234, max_iter=10000))
])

# models to test with
models = [
    ("LinearSVC", linear_svc),
    ("Ridge", ridge)
]

for model_name, model in models:
    for choice in [True, False]:
        model.set_params(**{"model__fit_intercept": choice})
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1234)
        scores = cross_validate(model, X, y, scoring=["accuracy", "f1_macro"], cv=cv, n_jobs=-1)
        logger.info(f"{model_name} with fit_intercept={choice}:")
        logger.info(f"Accuracy: {np.mean(scores['test_accuracy'])}")
        logger.info(f"F1: {np.mean(scores['test_f1_macro'])}")
