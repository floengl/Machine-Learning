from utils import load_training_dataset, setup_logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
import numpy as np

# set up logging
logger = setup_logging("test_ridge_dim_reduction")

# load dataset
X, y = load_training_dataset()

# max abs scaler
preprocessor = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("max_abs_scaler", MaxAbsScaler())
])

test = {
    "PCA": [0.9, 0.95, 0.99],
    "TSVD": [400, 500, 600, 700]
}

for name, n_components in test.items():
    for n in n_components:
        if name == "PCA":
            dim_reduction = PCA(n_components=n)
        elif name == "TSVD":
            dim_reduction = TruncatedSVD(n_components=n)
        else:
            ValueError("Invalid dimensionality reduction method")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("dim_reduction", dim_reduction),
            ("model", RidgeClassifier(random_state=1234))
        ])
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1234)
        scores = cross_validate(pipeline, X, y, scoring=["accuracy", "f1_macro"], cv=cv, n_jobs=-1)
        logger.info(f"{dim_reduction}:")
        logger.info("Accuracy: %.6f (%.6f)" % (np.mean(scores["test_accuracy"]), np.std(scores["test_accuracy"])))
        logger.info("F1: %.6f (%.6f)" % (np.mean(scores["test_f1_macro"]), np.std(scores["test_f1_macro"])))
