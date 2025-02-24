from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from utils import load_dataset, setup_logging
from utils import logger
import os
from utils.config import Config
from datetime import datetime

from simulated_annealing_2 import *

logger = setup_logging("search_2")

random_seed = 42
alpha = 0.9
maxtime = 180
mintime = 120
f = 5
n_repeats = 10
T_0 = 500
update_iters = 5
go_to_best_multiple = 4

models = {"rf": RandomForestRegressor,
          "LinearSVR": LinearSVR,
          "Ridge": Ridge,
          "Lasso": Lasso,
          "KNN": KNeighborsRegressor}


start_params = {"rf": {"n_estimators": 100, "max_depth": 10, "max_features": "log2"},
                "LinearSVR": {"max_iter": 1000, "C": 1.0, "fit_intercept": True},
                "Ridge": {"alpha": 1.0, "fit_intercept": True},
                "Lasso": {"alpha": 1.0, "fit_intercept": True},
                "KNN": {"n_neighbors": 5, "p": 5}}

params_vals = {"rf": {"n_estimators": [10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,500,600,700,800,900], "max_depth": [5,10,20,30,50,75,100,150,200],
                      "max_features": ["sqrt", "log2", None]},
                "LinearSVR": {"max_iter": [1000,2000,5000,10000], "C": [10**i for i in range(-3, 4)], "fit_intercept": [True, False]},
                "Ridge": {"alpha": [10**i for i in range(-3, 4)], "fit_intercept": [True, False]},
                "Lasso": {"alpha": [10**i for i in range(-3, 4)], "fit_intercept": [True, False]},
                "KNN": {"n_neighbors": [3,5,10,20], "p": [1,2,3,4,5]}}

X, Y = load_dataset()


model, results, nr_reheats, extime, T, best_i, best_time = simulate_annealing(start_params, params_vals, X, Y, models =  models, train_model=train_model_2, maxiters=10000000, T_0=T_0, f=f, n_repeats=n_repeats, random_seed=random_seed, alpha=alpha, mintime=mintime, maxtime=maxtime, update_iters=update_iters, go_to_best_multiple=go_to_best_multiple)

# Use the RESULTS_DIR from the config file
results_dir = Config.RESULTS_DIR
os.makedirs(results_dir, exist_ok=True)

# Get the current timestamp and format it
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create the results filename with the timestamp
results_file = os.path.join(results_dir, f"results_{timestamp}.csv")
results.to_csv(results_file, index=False)


logger.info(f"Random Seed: {random_seed}")
logger.info(f"folds: {f}, repeats: {n_repeats}")
logger.info(f"Alpha: {alpha}")
logger.info(f"Max time: {maxtime}") 
logger.info(f"Min time: {mintime}")
logger.info(f"Decrease temperature after {update_iters} iterations")
logger.info(f"Go to best model after {go_to_best_multiple*update_iters} iterations")

logger.info(F"Inital temperature: {T_0}\n\nFindings:\n")
logger.info(f"Best model: {model}")
logger.info(f"Best score: {results['Best Metric'].min()}")
logger.info(f"Number of reheats: {nr_reheats-1}")
logger.info(f"Execution time: {extime/60} minutes")
logger.info(f"Final temperature: {T}")
logger.info(f"Best iteration: {best_i}")
logger.info(f"Best results achieved after {best_time} minutes")
logger.info(f"Results saved to: {results_file}")



# Log the entire DataFrame
logger.info(f"History:\n{results}\n\n")
logger.info(f"Parameter values: {params_vals}")

print(model)
print(results)




