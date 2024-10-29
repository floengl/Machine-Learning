from utils import Config, load_training_dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.base import clone
import matplotlib.pyplot as plt
import numpy as np
import os

# load dataset
X, y = load_training_dataset()

# define estimator
estimator = Pipeline([
    ("preprocessor", MinMaxScaler()),
    ("model", RandomForestClassifier(random_state=1234, n_estimators=3000, max_depth=50,
                                     min_samples_split=6, min_samples_leaf=1, max_features=None))
])

# parameter ranges for sensitivity analysis
param_ranges = {
    "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "n_estimators": np.linspace(100, 3000, 10, dtype=int),
    "min_samples_split": range(2, 20),
    "min_samples_leaf": range(1, 6),
    "max_features": ["sqrt", "log2", None]
}

# plot sensitivity analysis
for param in param_ranges:
    print(f"Running sensitivity analysis on parameter {param}")
    accuracy = []
    recall = []
    for value in param_ranges[param]:
        model = clone(estimator)
        model.set_params(**{f"model__{param}": value})
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1234)
        scores = cross_validate(model, X, y, scoring=["accuracy", "recall"], cv=cv, n_jobs=-1)
        accuracy.append(scores["test_accuracy"].mean())
        recall.append(scores["test_recall"].mean())
    plt.figure(figsize=(7, 5))
    x_str = [str(value) for value in param_ranges[param]]
    if "None" in x_str or "True" in x_str or "False" in x_str:
        x = x_str
    else:
        x = param_ranges[param]
    plt.plot(x, accuracy, label="Accuracy")
    plt.plot(x, recall, label="Recall")
    plt.xlabel(param)
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Config.PLOTS_DIR, f"rf_{param}_sensitivity.pdf"))
