from utils import load_training_dataset, setup_logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pandas as pd

# Set up logging
logger = setup_logging("tune_linearsvc")

# load dataset
X, y = load_training_dataset()

# define preprocessing pipeline
preprocessor = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("max_abs_scaler", MaxAbsScaler())
])