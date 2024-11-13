from utils import Config, load_training_dataset, setup_logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# load dataset
X, y = load_training_dataset()

# define estimator
estimator = Pipeline([
    ("preprocessor", Pipeline([("none", "passthrough")])),
    ("model", LinearSVC(random_state=1234, max_iter=100000, dual=False))
])

class_weight = [None, "balanced"]
fit_intercept = [True, False]
C = 10**np.linspace(-6, 3, 50)

data = []
for cw in class_weight:
    for fi in fit_intercept:
        for c in C:
            estimator.set_params(model__class_weight=cw, model__fit_intercept=fi, model__C=c)
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1234)
            scores = cross_validate(estimator, X, y, scoring=["accuracy", "recall"], cv=cv, n_jobs=-1)
            data.append([str(cw), fi, c, scores["test_accuracy"].mean(), scores["test_recall"].mean()])


data = pd.DataFrame(data, columns=["class_weight", "fit_intercept", "C", "accuracy", "recall"])
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
for i, cw in enumerate(class_weight):
    for j, fi in enumerate(fit_intercept):
        df = data[(data["class_weight"] == str(cw)) & (data["fit_intercept"] == fi)]
        plot: plt.Axes = axs[i, j]
        plot.set_title(f"class_weight={cw}, fit_intercept={fi}")
        plot.semilogx(df["C"], df["accuracy"], label="accuracy")
        plot.semilogx(df["C"], df["recall"], label="recall")
        plot.set_xlabel("C")
        plot.set_ylabel("score")
        plot.set_ylim(0.32, 0.8)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2)
fig.tight_layout()
plt.savefig(os.path.join(Config.PLOTS_DIR, "linearsvc_parameter_sensitivity.pdf"))
plt.close()
# log scores
logger = setup_logging("linearsvc_parameter_sensitivity")
logger.info(data.to_string())
