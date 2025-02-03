import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error

# Custom scoring function for relative squared error
def relative_squared_error(y_true, y_pred):
    baseline_error = np.sum((y_true - np.mean(y_true)) ** 2)
    model_error = np.sum((y_true - y_pred) ** 2)
    return model_error / baseline_error

#make custom scorer
rse_scorer = make_scorer(relative_squared_error, greater_is_better=False)

mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)