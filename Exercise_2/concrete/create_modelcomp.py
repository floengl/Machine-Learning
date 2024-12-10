from utils import Config, load_dataset, setup_logging, rse_scorer, relative_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from random_forest import ourRandomForestRegressor
from llmrfr import LLMRandomForestRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, RepeatedKFold, train_test_split
import pandas as pd
import os
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

# load dataset
X, y = load_dataset()


# define estimator
ourrfr = Pipeline([
    ("preprocessor", RobustScaler()),
    ("model", ourRandomForestRegressor(random_state=1234, boot_type=False, max_depth=40, min_samples_split=2,max_features='log2', nb_samples='Full', nb_trees=40, max_workers=12))
])

# define estimator
sktrfr = Pipeline([
    ("preprocessor", RobustScaler()),
    ("model", RandomForestRegressor(random_state=1234, bootstrap=False, max_depth=30, min_samples_split=2,max_features='log2', n_estimators=300))
])

# define estimator
knn = Pipeline([
    ("preprocessor", RobustScaler()),
    ("model", KNeighborsRegressor(n_neighbors=5, weights='distance', p=5))
])

llmrfr = Pipeline([
    ("preprocessor", RobustScaler()),
    ("model", LLMRandomForestRegressor(random_state=1234, max_depth=None, min_samples_split=2, max_features=None, n_estimators=100))
])

# all models
models = [
    ("OurRFR", ourrfr),
    ("SKTRFR", sktrfr),
    ("KNN", knn),
    ("LLMRFR", llmrfr)
]

data_cv = []
data_holdout = []
data_runtime = []
plt.figure(figsize=(10, 6))
for name, model in models:
    # evaluate model with cross validation
    cv = RepeatedKFold(n_splits=4, n_repeats=3, random_state=1234)
    scores = cross_validate(model, X, y, scoring={"mse": "neg_mean_squared_error", "rse": rse_scorer}, cv=cv, n_jobs=-1)
    # evaluate model with holdout method and measure runtime
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    start = timer()
    model.fit(X_train, y_train)
    end = timer()
    train_time = end - start
    start = timer()
    y_pred = model.predict(X_test)
    end = timer()
    pred_time = end - start
    # save results
    data_runtime.append([name, train_time, pred_time])
    data_holdout.append([f"{name}\nMse", -mean_squared_error(y_test, y_pred)])
    data_holdout.append([f"{name}\nRse", -relative_squared_error(y_test, y_pred)])
    for fold in range(len(scores["test_mse"])):
        for score in ["mse", "rse"]:
            data_cv.append([f"{name}\n{score.capitalize()}", name, scores[f"test_{score}"][fold]])
df_cv = pd.DataFrame(data_cv, columns=["model_score", "model", "score"])
df_holdout = pd.DataFrame(data_holdout, columns=["model_score", "score"])
df_runtime = pd.DataFrame(data_runtime, columns=["model", "train_time", "pred_time"])


# Plot MSE
plt.figure(figsize=(10, 6))
meanlineprops = dict(linestyle='--', linewidth=1.5, color='white')
my_pal = {model_score: "C0" if "OurRFR" in model_score else "C1" if "SKTRFR" in model_score else "C2" if "KNN" in model_score else "C3"
          for model_score in df_cv["model_score"].unique()}
sns.set_theme(style="darkgrid")
sns.color_palette("pastel")
ax = sns.boxplot(x="model_score", y="score", data=df_cv[df_cv["model_score"].str.contains("Mse")], palette=my_pal, showmeans=True,
                 meanline=True, meanprops=meanlineprops)

# Plot holdout results as red crosses for MSE
holdout_x = df_holdout[df_holdout["model_score"].str.contains("Mse")]["model_score"].unique()
holdout_y = df_holdout[df_holdout["model_score"].str.contains("Mse")].groupby("model_score")["score"].mean().reindex(holdout_x)
plt_holdout, = plt.plot(holdout_x, holdout_y, "x", color="red", markersize=10, markeredgewidth=2, label="Holdout Result")

mean_line_dummy, = plt.plot([], [], '--', linewidth=1.5, color='white', label='Mean')
plt.legend(handles=[plt_holdout, mean_line_dummy], loc="lower right")
plt.xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(Config.PLOTS_DIR, "model_comparison_mse.pdf"))

# Plot RSE
plt.figure(figsize=(10, 6))
ax = sns.boxplot(x="model_score", y="score",hue = "model_score", data=df_cv[df_cv["model_score"].str.contains("Rse")], palette=my_pal, showmeans=True,
                 meanline=True, meanprops=meanlineprops)

# Plot holdout results as red crosses for RSE
holdout_x = df_holdout[df_holdout["model_score"].str.contains("Rse")]["model_score"].unique()
holdout_y = df_holdout[df_holdout["model_score"].str.contains("Rse")].groupby("model_score")["score"].mean().reindex(holdout_x)
plt_holdout, = plt.plot(holdout_x, holdout_y, "x", color="red", markersize=10, markeredgewidth=2, label="Holdout Result")

mean_line_dummy, = plt.plot([], [], '--', linewidth=1.5, color='white', label='Mean')
plt.legend(handles=[plt_holdout, mean_line_dummy], loc="lower right")
plt.xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(Config.PLOTS_DIR, "model_comparison_rse.pdf"))

# log results
logger = setup_logging("model_comparison")
logger.info("\nRuntime of models:")
logger.info(df_runtime)
logger.info("\nHoldout results:")
df_holdout["model_score"] = df_holdout["model_score"].str.replace("\n", " ")
logger.info(df_holdout)
logger.info("\nCross validation results:")
df_cv["model_score"] = df_cv["model_score"].str.replace("\n", " ")
logger.info(df_cv.groupby("model_score").agg({'score': ["mean", "std"]}))
