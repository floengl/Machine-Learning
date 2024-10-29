from utils import load_training_dataset
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD

# load dataset
X, y = load_training_dataset()

X.drop("ID", axis=1, inplace=True)
X = MaxAbsScaler().fit_transform(X)
tsvd = TruncatedSVD(n_components=9999)
tsvd.fit(X)

tsvd_var_ratios = tsvd.explained_variance_ratio_


def select_n_components(var_ratio, goal_var: float) -> int:
    total_variance = 0.0
    n_components = 0

    for explained_variance in var_ratio:
        total_variance += explained_variance
        n_components += 1
        if total_variance >= goal_var:
            break
    return n_components


print(select_n_components(tsvd_var_ratios, 0.90))
print(select_n_components(tsvd_var_ratios, 0.95))
print(select_n_components(tsvd_var_ratios, 0.99))
