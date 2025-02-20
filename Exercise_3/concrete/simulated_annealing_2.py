import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score, RepeatedKFold
import copy




# Sample train_model function #
###############################

def train_model_1(models, curr_model, curr_params, Xtrain, Xvalid, Ytrain, Yvalid):

     model = models[curr_model](**curr_params)
     model.fit(Xtrain, Ytrain)
     preds = model.predict(Xvalid)
     metric_val = root_mean_squared_error(Yvalid, preds) # Any metric can be used

     return model, metric_val


def train_model_2(models, curr_model, curr_params, X, Y, f=5, n_repeats=1, random_seed=42):
    np.random.seed(random_seed)
    model = models[curr_model](**curr_params)
    rkf = RepeatedKFold(n_splits=f, n_repeats=n_repeats, random_state=random_seed)
    cv_scores = cross_val_score(model, X, Y, cv=rkf, scoring='neg_root_mean_squared_error')
    mean_cv_score = -cv_scores.mean()  # Convert negative RMSE to positive RMSE

    return model, mean_cv_score


def choose_params(curr_model, params_vals, curr_params=None, T=400, T_0=400, random_seed=42):
    np.random.seed(random_seed)
    if curr_params[curr_model] is not None:
        next_params = copy.deepcopy(curr_params[curr_model])
        param_to_update = np.random.choice(list(params_vals[curr_model].keys()))
        curr_index = params_vals[curr_model][param_to_update].index(curr_params[curr_model][param_to_update])

        # Determine the new index using a normal distribution around the current index
        std_dev = max(1, int(len(params_vals[curr_model][param_to_update]) * T / (2 * T_0)))
        new_index = int(np.random.normal(loc=curr_index, scale=std_dev))
        new_index = max(0, min(len(params_vals[curr_model][param_to_update]) - 1, new_index))  # Ensure index is within bounds

        next_params[param_to_update] = params_vals[curr_model][param_to_update][new_index]
    else:
        next_params = {k: np.random.choice(v) for k, v in params_vals[curr_model].items()}

    return next_params


def choose_model(models, best_model=None, prev_model=None, T=400, T_0=400, go_to_best_model=False, random_seed=42):
  
    np.random.seed(random_seed)
    if go_to_best_model:
        return best_model
    non_switch_probability = 1 - np.exp(-0.25 * (T_0 / T) ** 2)  # starts with a higher switch probability and decreases faster than acceptance probability for worse solutions due to squaring the value
    if np.random.rand() < non_switch_probability and prev_model is not None:
        return prev_model
    else:
        return np.random.choice(list(models.keys()))
    


def simulate_annealing(start_params, param_vals, X, Y,  models, train_model=train_model_2, maxiters=100, alpha=0.9, beta=1.3, T_0=400, update_iters=5, f=5, n_repeats=1, random_seed=42):

    rng = np.random.RandomState(random_seed)
    columns = ['Model'] + [f"{model}_{param}" for model in models for param in start_params[model].keys()] + ['Metric', 'Best Metric']
    results = pd.DataFrame(index=range(maxiters), columns=columns)
    best_metric = float('inf')
    prev_metric = float('inf')
    prev_params = copy.deepcopy(start_params)
    best_params = copy.deepcopy(start_params)
    prev_model = None
    best_model = None

    go_to_best_model= False
    T = T_0

    for i in range(maxiters):
        print('Starting Iteration {}'.format(i))

        curr_seed = rng.randint(0, 1000000)
        curr_model = choose_model(models, best_model, prev_model, T, T_0, go_to_best_model, random_seed=curr_seed)
        curr_seed = rng.randint(0, 1000000)
        curr_params = choose_params(curr_model, param_vals, prev_params, T, T_0, random_seed=curr_seed)
        print('Model: {} Parameters: {}'.format(curr_model, curr_params))

        curr_seed = rng.randint(0, 1000000)
        model, metric = train_model(models, curr_model, curr_params, X, Y, f=f, n_repeats=n_repeats, random_seed=curr_seed)

        if metric < prev_metric:
            print('Local Improvement in metric from {:8.4f} to {:8.4f} '
                  .format(prev_metric, metric) + ' - parameters accepted')
            prev_model = curr_model
            prev_params[curr_model] = copy.deepcopy(curr_params)
            prev_metric = metric

            if metric < best_metric:
                print('Global improvement in metric from {:8.4f} to {:8.4f} '
                      .format(best_metric, metric) +
                      ' - best parameters updated')
                best_model = curr_model
                best_metric = metric
                best_params[curr_model] = copy.deepcopy(curr_params)
                best_model_final = model

        else:
            rnd = rng.uniform()
            diff = metric - prev_metric
            threshold = np.exp(-beta * diff / T)
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

         # Store the model name
        results.loc[i, 'Model'] = curr_model

        # Store the hyperparameters for the current model
        for param, value in curr_params.items():
            results.loc[i, f"{curr_model}_{param}"] = value

        # Store the metrics
        results.loc[i, 'Metric'] = metric
        results.loc[i, 'Best Metric'] = best_metric

        go_to_best_model = False
        if i % update_iters == 0:
            T = alpha * T
        if i % update_iters*6 == 0:
            go_to_best_model = True

    return  best_model_final, results