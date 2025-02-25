from utils import Config, load_dataset, rse_scorer
from sklearn.neighbors import KNeighborsRegressor
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
        ("model", KNeighborsRegressor( n_neighbors=5, weights='distance', p=5))
    ])

    # search space
    param_ranges = {
        "n_neighbors":  np.linspace(1, 50, dtype=int),
        "weights": ['uniform', 'distance'],
        "p": np.linspace(1, 5, dtype=int),
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

        ax1.plot(x, mse, 'b-', label="MSE")
        ax1.set_xlabel(param)
        ax1.set_ylabel("MSE", color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        ax2 = ax1.twinx()
        ax2.plot(x, rse, label="RSE", color = 'orange')
        ax2.set_ylabel("RSE", color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        fig.tight_layout()
        plt.title(f"Sensitivity Analysis for {param}")
        plt.savefig(os.path.join(Config.PLOTS_DIR, f"knn_{param}_sensitivity_2scale.pdf"))

if __name__ == "__main__":
    main()