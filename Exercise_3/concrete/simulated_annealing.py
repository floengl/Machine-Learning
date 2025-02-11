import pandas as pd
import numpy as np
from collections import OrderedDict
from random import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import root_mean_squared_error
from utils import load_dataset, setup_logging
from utils import logger
import copy




# Sample train_model function #
###############################

def train_model(models, curr_model,curr_params, start_param, Xtrain, Xvalid, Ytrain,
                 Yvalid):
     """
     Train the model with given set of hyperparameters
     curr_params - Dict of hyperparameters and chosen values
     param - Dict of hyperparameters that are kept constant
     Xtrain - Train Data
     Xvalid - Validation Data
     Ytrain - Train labels
     Yvalid - Validaion labels
     """


     
     params_copy = start_param[curr_model].copy()
     params_copy.update(curr_params)
     model = models[curr_model](**params_copy)
     model.fit(Xtrain, Ytrain)
     preds = model.predict(Xvalid)
     metric_val = root_mean_squared_error(Yvalid, preds) # Any metric can be used

     return model, metric_val


def choose_params(curr_model,start_params, params_vals, curr_params=None, T=0.4):

    print(curr_model)
    print(curr_params[curr_model])
    if curr_params[curr_model] is not None:
        next_params = copy.deepcopy(curr_params[curr_model])
        param_to_update = np.random.choice(list(start_params[curr_model].keys()))
        param_vals = curr_params[curr_model][param_to_update]
        curr_index = params_vals[curr_model][param_to_update].index(curr_params[curr_model][param_to_update])
        
        # Determine the new index using a normal distribution around the current index
        std_dev = max(1, int(len(params_vals[curr_model][param_to_update]) * T))
        new_index = int(np.random.normal(loc=curr_index, scale=std_dev))
        new_index = max(0, min(len(params_vals[curr_model][param_to_update]) - 1, new_index))  # Ensure index is within bounds
        
        next_params[param_to_update] = params_vals[curr_model][param_to_update][new_index]
    else:
        next_params = {k: np.random.choice(v) for k, v in start_params[curr_model].items()}

    return next_params


def choose_model(models, best_model= None, prev_model=None, T=0.4, T_0=0.4, go_to_best_model=False):

    """
    Function to choose model for next iteration
    Output:
    String of model name
    """
    if go_to_best_model:
        return best_model
    switch_probabilty = 0.5* T/T_0
    if random() < switch_probabilty and prev_model is not None:
        return prev_model
    else:
        return np.random.choice(list(models.keys()))
    


def simulate_annealing(start_params,
                       param_vals,
                       X_train,
                       X_valid,
                       Y_train,
                       Y_valid,
                       train_model,
                       models,
                       maxiters=100,
                       alpha=0.85,
                       beta=1.3,
                       T_0=0.40,
                       update_iters=5):
    """
    Function to perform hyperparameter search using simulated annealing
    Inputs:
    start_params - Ordered dictionary of Hyperparameter search space
    const_param - Static parameters of the model
    Xtrain - Train Data
    Xvalid - Validation Data
    Ytrain - Train labels
    Yvalid - Validaion labels
    fn_train - Function to train the model
        (Should return model and metric value as tuple, sample commented above)
    maxiters - Number of iterations to perform the parameter search
    alpha - factor to reduce temperature
    beta - constant in probability estimate
    T_0 - Initial temperature
    update_iters - # of iterations required to update temperature
    Output:
    Dataframe of the parameters explored and corresponding model performance
    """
    columns = [*start_params.keys()] + ['Metric', 'Best Metric']
    results = pd.DataFrame(index=range(maxiters), columns=columns)
    best_metric = -1.
    prev_metric = -1.
    prev_params = copy.deepcopy(start_params)
    best_params = dict()
    prev_model = None
    best_model = None

    go_to_best_model= False
    T = T_0

    for i in range(maxiters):
        print('Starting Iteration {}'.format(i))


        curr_model = choose_model(models,best_model,prev_model, T, T_0, go_to_best_model)
        print(prev_params[curr_model])
        curr_params = choose_params(curr_model, start_params, param_vals, prev_params, T)


        model, metric = train_model(models, curr_model, curr_params, start_params, X_train,
                                 X_valid, Y_train, Y_valid)

        if metric > prev_metric:
            print('Local Improvement in metric from {:8.4f} to {:8.4f} '
                  .format(prev_metric, metric) + ' - parameters accepted')
            prev_model = curr_model
            prev_params[curr_model] = copy.deepcopy(curr_params)
            prev_metric = metric

            if metric > best_metric:
                print('Global improvement in metric from {:8.4f} to {:8.4f} '
                      .format(best_metric, metric) +
                      ' - best parameters updated')
                best_model = curr_model
                best_metric = metric
                best_params[curr_model] = copy.deepcopy(curr_params)
                best_model_final = model

        else:
            rnd = np.random.uniform()
            diff = metric - prev_metric
            threshold = np.exp(beta * diff / T)
            if rnd < threshold:
                print('No Improvement but parameters accepted. Metric change' +
                      ': {:8.4f} threshold: {:6.4f} random number: {:6.4f}'
                      .format(diff, threshold, rnd))
                prev_metric = metric
                prev_params[curr_model] = copy.deepcopy(curr_params)
            else:
                print('No Improvement and parameters rejected. Metric change' +
                      ': {:8.4f} threshold: {:6.4f} random number: {:6.4f}'
                      .format(diff, threshold, rnd))

        results.loc[i, list(curr_params.keys())] = list(curr_params.values())
        results.loc[i, 'Metric'] = metric
        results.loc[i, 'Best Metric'] = best_metric

        go_to_best_model = False
        if i % update_iters == 0:
            T = alpha * T
            go_to_best_model = True

    return  best_model_final