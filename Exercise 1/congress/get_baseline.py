from utils import load_training_dataset, setup_logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
import numpy as np

# Set up logging
logger = setup_logging("get_baseline")

# load dataset
X, y = load_training_dataset()

# dummy classifier pipeline
dummy = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("one_hot_encoder", OneHotEncoder(drop="if_binary")),
    ("dummy", DummyClassifier(strategy="most_frequent"))
])

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1234)
scores = cross_validate(dummy, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
logger.info(f"Accuracy: {np.mean(scores['test_score'])} +/- {np.std(scores['test_score'])}")
