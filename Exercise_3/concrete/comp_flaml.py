from flaml import AutoML, tune
from utils import load_dataset
from utils.config import Config
from flaml.automl.model import SKLearnEstimator
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold


class MyLasso(SKLearnEstimator):
    def __init__(self, task="regression", n_jobs = None, **config):
        super().__init__(task, **config)

        self.estimator_class = Lasso

    @classmethod
    def search_space(cls, data_size, task):
        space = {
        "alpha": {
            "domain": tune.choice([10**i for i in range(-3, 4)]),
        },
        "fit_intercept": {
            "domain": tune.choice([True, False]),
        }
        }
        return space

class MyRidge(SKLearnEstimator):
    def __init__(self, task="regression", n_jobs = None, **config):
        super().__init__(task, **config)

        self.estimator_class = Ridge

    @classmethod
    def search_space(cls, data_size, task):
        space = {
        "alpha": {
            "domain": tune.choice([10**i for i in range(-3, 4)]),
        },
        "fit_intercept": {
            "domain": tune.choice([True, False]),
        }
        }
        return space
    
class MyLinearSVR(SKLearnEstimator):
    def __init__(self, task="regression", n_jobs = None, **config):
        super().__init__(task, **config)

        self.estimator_class = LinearSVR

    @classmethod
    def search_space(cls, data_size, task):
        space = {
        "max_iter": {
            "domain": tune.choice([1000,2000,5000,10000]),
        },
        "C": {
            "domain": tune.choice([10**i for i in range(-3, 4)]),
        },
        "fit_intercept": {
            "domain": tune.choice([True, False]),
        },
        }
        return space

class MyKNeighborsRegressor(SKLearnEstimator):
    def __init__(self, task="regression", **config):
        super().__init__(task, **config)

        self.estimator_class = KNeighborsRegressor

    @classmethod
    def search_space(cls, data_size, task):
        space = {
        "n_neighbors": {
            "domain": tune.choice([3,5,10,20]),
        },
        "p": {
            "domain": tune.choice([1,2,3,4,5]),
        }
        }
        return space

class MyRF(SKLearnEstimator):
    def __init__(self, task="regression", **config):
        super().__init__(task, **config)

        self.estimator_class = RandomForestRegressor

    @classmethod
    def search_space(cls, data_size, task):
        space = {
        "n_estimators": {
            "domain": tune.choice([10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,500,600,700,800,900]),
        },
        "max_depth": {
            "domain": tune.choice([5,10,20,30,50,75,100,150,200]),
        },
        "max_features": {
            "domain": tune.choice(['sqrt', 'log2', None]),
        }
        }
        return space



X, y = load_dataset()

automl = AutoML()

automl.add_learner("MyKNN", MyKNeighborsRegressor)
automl.add_learner("MyRF", MyRF)
automl.add_learner("MyLinearSVR", MyLinearSVR)
automl.add_learner("MyRidge", MyRidge)
automl.add_learner("MyLasso", MyLasso)

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

automl_settings = {
    "time_budget": 5400,
    "eval_method": "cv",
    "split_type": rkf,
    "estimator_list": ["MyKNN", "MyRF", "MyLinearSVR", "MyRidge", "MyLasso"],
    "metric": "rmse",
    "task": "regression",
    "log_file_name": "comp_flaml_concrete.log",
    "seed": 42
}

automl.fit(X_train=X, y_train=y, **automl_settings)