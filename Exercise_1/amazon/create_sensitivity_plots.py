from utils import Config, load_training_dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.base import clone
import matplotlib.pyplot as plt
import numpy as np
import os

# load dataset
X, y = load_training_dataset()

# define LinearSVC pipeline
linear_svc = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("power_transform", PowerTransformer()),
    ("model", LinearSVC(random_state=1234, max_iter=10000, class_weight="balanced", C=0.00946,dual=True, fit_intercept=True))
])

# define RidgeClassifier pipeline
ridge = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("power_transformer", PowerTransformer()),
    ("pca", PCA(n_components=0.99)),
    ("model", RidgeClassifier(random_state=1234, alpha=45.32, fit_intercept=False, class_weight=None))
])

# define Random Forest pipeline
random_forest = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("powertransform", PowerTransformer()),
    ("model", RandomForestClassifier(random_state=1234, n_estimators=2000, max_depth=50,
                                     min_samples_split=2, min_samples_leaf=1, max_features="sqrt"))
])

# all models
models = [
    #("LinearSVC", linear_svc),
    ("RidgeClassifier", ridge),
    ("RandomForestClassifier", random_forest)
]

# parameter ranges for sensitivity analysis
param_ranges = {
    "LinearSVC": {
        "C": 10**np.linspace(-3, 3, 15),
        "class_weight": [None, "balanced"],
        "fit_intercept": [True, False],
        "dual": [True, False]
    },
    "RidgeClassifier": {
        "alpha": np.linspace(0, 100, 20),
        "fit_intercept": [True, False],
        "class_weight": [None, "balanced"]
    },
    "RandomForestClassifier": {
        "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "n_estimators": np.linspace(100, 2000, 10, dtype=int),
        "min_samples_split": range(2, 15),
        "min_samples_leaf": range(1, 6),
        "max_features": ["sqrt", "log2"]
    }
}

# plot sensitivity analysis
for name, pipeline in models:
    for param in param_ranges[name]:
        print(f"Running sensitivity analysis for {name} on parameter {param}")
        accuracy = []
        f1 = []
        for value in param_ranges[name][param]:
            model = clone(pipeline)
            model.set_params(**{f"model__{param}": value})
            n_repeats = 1 if name == "RandomForestClassifier" else 1
            cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=n_repeats, random_state=1234)
            scores = cross_validate(model, X, y, scoring=["accuracy", "f1_macro"], cv=cv, n_jobs=-1)
            accuracy.append(scores["test_accuracy"].mean())
            f1.append(scores["test_f1_macro"].mean())
        plt.figure(figsize=(7, 5))
        x_str = [str(value) for value in param_ranges[name][param]]
        if "None" in x_str or "True" in x_str or "False" in x_str:
            x = x_str
        else:
            x = param_ranges[name][param]
        if param == "C":
            plt.semilogx(x, accuracy, label="Accuracy")
            plt.semilogx(x, f1, label="F1 Macro")
        else:
            plt.plot(x, accuracy, label="Accuracy")
            plt.plot(x, f1, label="F1 Macro")
        plt.xlabel(param)
        plt.ylabel("score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(Config.PLOTS_DIR, f"{name}_{param}_sensitivity.pdf"))
