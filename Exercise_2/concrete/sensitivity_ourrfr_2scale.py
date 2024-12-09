from utils import Config, load_dataset, rse_scorer
from random_forest import ourRandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.base import clone
import os

def main():

    X, y = load_dataset()

    # define estimator
    estimator = Pipeline([
        ("preprocessor", RobustScaler()),
        ("model", ourRandomForestRegressor(random_state=1234, boot_type=False, max_depth=40, min_samples_split=2,max_features='log2', nb_samples='Full', nb_trees=40))
    ])

    # search space
    param_ranges = {
        "nb_trees": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300],
        "max_depth": [-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "min_samples_split": np.linspace(2, 20, 1, dtype=int),
        "nb_samples": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, "Full"],
        "max_features": [None, "sqrt", "log2"],
        "boot_type": [True, False],
    }

    # plot sensitivity analysis
    for param in param_ranges:
        print(f"Running sensitivity analysis on parameter {param}")
        rse = []
        mse = []
        for value in param_ranges[param]:
            model = clone(estimator)
            model.set_params(**{f"model__{param}": value})
            cv = RepeatedKFold(n_splits=4, n_repeats=3, random_state=1234)
            scores = cross_validate(model, X, y, scoring={"neg_mean_squared_error": "neg_mean_squared_error", "rse": rse_scorer}, cv=cv, n_jobs=-1)
            mse.append(scores["test_neg_mean_squared_error"].mean())
            rse.append(scores["test_rse"].mean())
        fig, ax1 = plt.subplots(figsize=(7, 5))

        x_str = [str(value) for value in param_ranges[param]]
        if "None" in x_str or "True" in x_str or "False" in x_str:
            x = x_str
        else:
            x = param_ranges[param]

        ax1.plot(x, mse, 'cyan-', label="MSE")
        ax1.set_xlabel(param)
        ax1.set_ylabel("MSE", color='cyan')
        ax1.tick_params(axis='y', labelcolor='cyan')

        ax2 = ax1.twinx()
        ax2.plot(x, rse, 'orange-', label="RSE")
        ax2.set_ylabel("RSE", color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        fig.tight_layout()
        plt.title(f"Sensitivity Analysis for {param}")
        plt.savefig(os.path.join(Config.PLOTS_DIR, f"knn_{param}_sensitivity_2scale.pdf"))

if __name__ == "__main__":
    main()