from utils import Config, load_training_dataset, load_test_dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
from datetime import datetime

# load dataset
X, y, le = load_training_dataset(return_label_encoder=True)

# load test dataset
X_test = load_test_dataset()

# define LinearSVC pipeline
linear_svc = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("one_hot_encoder", OneHotEncoder(drop="if_binary")),
    ("model", LinearSVC(random_state=1234, max_iter=100000, C=1.2,fit_intercept=True, class_weight="balanced", dual=False))
])

# define RidgeClassifier pipeline
ridge = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("one_hot_encoder", OneHotEncoder(drop="if_binary")),
    ("model", RidgeClassifier(random_state=1234, alpha=0.0, fit_intercept=True, class_weight=None))
])

# define Random Forest pipeline
random_forest = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("one_hot_encoder", OneHotEncoder(drop="if_binary")),
    ("model", RandomForestClassifier(random_state=1234, n_estimators=1963, max_depth=40,
                                     min_samples_split=20, min_samples_leaf=4, max_features=None))
])

# all models
models = [
    ("LinearSVC", linear_svc),
    ("RidgeClassifier", ridge),
    ("RandomForestClassifier", random_forest)
]

# fit models
for name, model in models:
    model.fit(X, y)
    y_pred = model.predict(X_test)
    # combine ID and prediction
    df = pd.DataFrame({"ID": X_test["ID"], "Class": le.inverse_transform(y_pred)})
    path = os.path.join(Config.SUBMISSION_DIR, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(path, index=False)
