import numpy as np
from utils import load_dataset
from sklearn.model_selection import train_test_split


class RegressionTree:

    class Node:
        def __init__(self, col=-1, threshold=None, result = None, left=None, right=None):
            self.col = col
            self.threshold = threshold
            self.result = result
            self.left = left
            self.right = right

    def __init__(self, max_depth=-1, min_samples_split=2, max_features=None, random_state=None):
        self.root_node = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.features_indexes = []

    def build_tree(self, X, y, score, depth):
        if len(y) < self.min_samples_split:
            # print("Not enough samples to split")
            return self.Node(result=np.mean(y))
        if depth == 0:
            # print("Max depth reached")
            return self.Node(result=np.mean(y))
        best_feature, best_threshold, best_score = None, None, float('inf')
        dim=X.shape[1]
        if self.max_features==None:
            sel_features=range(dim)
        elif self.max_features=="sqrt":
            sel_features=self.rng.choice(range(dim), int(np.sqrt(dim)))
        elif self.max_features=="log2":
            sel_features=self.rng.choice(range(dim),int(np.log2(dim)))
        elif isinstance(self.max_features, int):
            sel_features=self.rng.choice(range(dim),self.max_features)
        for feature in sel_features:
            feature_values = X[:, feature]
            for threshold in np.unique(feature_values):
                y_left = y[feature_values < threshold]
                y_right = y[feature_values >= threshold]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                current_score = score(y, y_left, y_right)
                if current_score < best_score:
                    best_score = current_score
                    best_threshold = threshold
                    best_feature = feature
        if best_threshold is None:
            # print("No best threshold found")
            return self.Node(result=np.mean(y))
        else:
            left = self.build_tree(X[X[:, best_feature] < best_threshold], y[X[:, best_feature] < best_threshold], score, depth - 1)
            right = self.build_tree(X[X[:, best_feature] >= best_threshold], y[X[:, best_feature] >= best_threshold], score, depth - 1)
            return self.Node(col=best_feature, threshold=best_threshold, left=left, right=right)
    
    def mse(self, y, y_left, y_right):
        return (len(y_left) / len(y)) * np.mean((y_left - np.mean(y_left)) ** 2) + (len(y_right) / len(y)) * np.mean((y_right - np.mean(y_right)) ** 2)
    
    def std(self, y, y_left, y_right):
        return (len(y_left) / len(y)) * np.std(y_left) + (len(y_right) / len(y)) * np.std(y_right)
    
    def fit(self, X, y, score=None):       
        if not score:
            score = self.std

        self.root_node = self.build_tree(X, y, score, self.max_depth)

    def regress(self, observation, node):
        if node.result is not None:
            return node.result
        else:
            val = observation[node.col]

            branch = None
            if val < node.threshold:
                branch = node.left
            else:
                branch = node.right
            return self.regress(observation, branch)
        
    def predict(self, features):

        y_pred = [self.regress(row, self.root_node) for row in features]
        return y_pred


if __name__ == "__main__":
    X, y = load_dataset()
    X = X
    y = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

    tree = RegressionTree(max_depth=6, min_samples_split=2, max_features="log2", random_state=1234)
    tree.fit(X_train.values, y_train.values)
    y_pred = tree.predict(X_test.values)
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"MSE: {mse}")