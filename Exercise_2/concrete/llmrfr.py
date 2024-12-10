import numpy as np
from utils import load_dataset
from sklearn.model_selection import train_test_split

# Custom Decision Tree for Regression with Standard Deviation as the cost function
import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth=0):
        if len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return np.mean(y)

        best_split = self._find_best_split(X, y)
        if best_split is None:
            return np.mean(y)

        left_idx = best_split['left_idx']
        right_idx = best_split['right_idx']
        left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return {
            'feature_idx': best_split['feature_idx'],
            'threshold': best_split['threshold'],
            'left': left_subtree,
            'right': right_subtree,
        }

    def _find_best_split(self, X, y):
        best_split = None
        best_cost = float('inf')

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_idx = np.where(X[:, feature_idx] <= threshold)[0]
                right_idx = np.where(X[:, feature_idx] > threshold)[0]

                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue

                cost = self._compute_split_cost(y[left_idx], y[right_idx])
                if cost < best_cost:
                    best_cost = cost
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'left_idx': left_idx,
                        'right_idx': right_idx,
                    }

        return best_split

    def _compute_split_cost(self, left_y, right_y):
        left_std = np.std(left_y) * len(left_y)
        right_std = np.std(right_y) * len(right_y)
        return left_std + right_std

    def _predict_one(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        feature_idx = tree['feature_idx']
        threshold = tree['threshold']
        if x[feature_idx] <= threshold:
            return self._predict_one(x, tree['left'])
        else:
            return self._predict_one(x, tree['right'])


class LLMRandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_subsets = []
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        # Convert to NumPy arrays
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        # Interpret `max_features`
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                self.max_features = int(np.sqrt(n_features))
            elif self.max_features == "log2":
                self.max_features = int(np.log2(n_features))
            else:
                raise ValueError(f"Invalid value for max_features: {self.max_features}")
        elif self.max_features is None:
            self.max_features = n_features
        elif isinstance(self.max_features, (int, float)):
            self.max_features = int(self.max_features)

        for _ in range(self.n_estimators):
            # Sample data with replacement (Bootstrap)
            indices = self.rng.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            # Randomly select feature subset
            feature_indices = self.rng.choice(n_features, self.max_features, replace=False)
            self.feature_subsets.append(feature_indices)

            # Train a decision tree on the bootstrap sample
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample[:, feature_indices], y_sample)
            self.trees.append(tree)

    def predict(self, X):
        X = np.array(X)
        predictions = np.array([
            tree.predict(X[:, feature_indices])
            for tree, feature_indices in zip(self.trees, self.feature_subsets)
        ])
        return np.mean(predictions, axis=0)
    
    # only thing implemented ourselves so that we can tune it
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for estimator {self.__class__.__name__}.")
        return self


# Example usage
if __name__ == "__main__":
    X, y = load_dataset()
    X = X
    y = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

    tree = DecisionTreeRegressor()
    tree.fit(X_train.values, y_train.values)
    y_pred = tree.predict(X_test.values)
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"MSE of individual tree: {mse}")

    rf = LLMRandomForestRegressor(n_estimators=40, max_depth=5)
    rf.fit(X_train.values, y_train.values)
    y_pred = rf.predict(X_test.values)
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"MSE of RF: {mse}")