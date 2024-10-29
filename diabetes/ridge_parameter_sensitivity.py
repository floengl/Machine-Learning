from utils import Config, load_training_dataset, setup_logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.linear_model import RidgeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# load dataset
X, y = load_training_dataset()

# define estimator
estimator = Pipeline([
    ("preprocessor", StandardScaler()),
    ("model", RidgeClassifier(random_state=1234))
])

fit_intercept = [True, False]
alpha = np.linspace(0, 100, 50)

data = []
for fi in fit_intercept:
    for a in alpha:
        estimator.set_params(model__fit_intercept=fi, model__alpha=a)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1234)
        scores = cross_validate(estimator, X, y, scoring=["accuracy", "recall"], cv=cv, n_jobs=-1)
        data.append([fi, a, scores["test_accuracy"].mean(), scores["test_recall"].mean()])


data = pd.DataFrame(data, columns=["fit_intercept", "alpha", "accuracy", "recall"])
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for j, fi in enumerate(fit_intercept):
    df = data[(data["fit_intercept"] == fi)]
    plot: plt.Axes = axs[j]
    plot.set_title(f"fit_intercept={fi}")
    plot.plot(df["alpha"], df["accuracy"], label="accuracy")
    plot.plot(df["alpha"], df["recall"], label="recall")
    plot.set_xlabel("alpha")
    plot.set_ylabel("score")
    plot.set_ylim(0.5, 0.8)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2)
fig.tight_layout()
plt.savefig(os.path.join(Config.PLOTS_DIR, "ridge_parameter_sensitivity.pdf"))
plt.close()
# log scores
logger = setup_logging("ridge_parameter_sensitivity")
logger.info(data.to_string())
