from utils import load_training_dataset, setup_logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import numpy as np
import pandas as pd

# Set up logging
logger = setup_logging("test_linearscv_svc")

# load dataset
X, y = load_training_dataset()

# define preprocessing pipeline
preprocessor = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough"))#,
    #("max_abs_scaler", MaxAbsScaler())
])

# models to test with
models = [
    ("LinearSVC", LinearSVC(random_state=1234)),
    ("SVC", SVC(random_state=1234,kernel="linear"))
]

# Run the two models
for model_name, model in models:
    accuracy = []
    f1 = []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1234)
    scores = cross_validate(Pipeline([("scaler",preprocessor),("model",model)]), X, y, scoring=["accuracy", "f1_macro"], cv=cv, n_jobs=-1)
    accuracy.append((np.mean(scores['test_accuracy']), np.std(scores['test_accuracy'])))
    f1.append((np.mean(scores['test_f1_macro']), np.std(scores['test_f1_macro'])))
    logger.info(f"\nAccuracy {model_name}:")
    logger.info(pd.DataFrame(accuracy, columns=['mean', 'std']).sort_values(by='mean', ascending=False))
    logger.info(f"\nF1 {model_name}:")
    logger.info(pd.DataFrame(f1, columns=['mean', 'std']).sort_values(by='mean', ascending=False))