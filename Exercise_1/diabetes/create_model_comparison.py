from utils import Config, load_training_dataset, setup_logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold, train_test_split
import pandas as pd
import os
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score

# load dataset
X, y = load_training_dataset()

# define LinearSVC pipeline
linear_svc = Pipeline([
    ("preprocessor", StandardScaler()),
    ("model", LinearSVC(random_state=1234, max_iter=100000, dual=False, C=0.028072,
                        class_weight="balanced", fit_intercept=False))
])

# define RidgeClassifier pipeline
ridge = Pipeline([
    ("preprocessor", StandardScaler()),
    ("model", RidgeClassifier(random_state=1234, alpha=40, fit_intercept=False))
])

# define Random Forest pipeline
random_forest = Pipeline([
    ("preprocessor", MinMaxScaler()),
    ("model", RandomForestClassifier(random_state=1234, n_estimators=3000, max_depth=50,
                                     min_samples_split=6, min_samples_leaf=1, max_features=None))
])

# all models
models = [
    ("LinearSVC", linear_svc),
    ("Ridge", ridge),
    ("RandomForest", random_forest)
]

data_cv = []
data_holdout = []
data_runtime = []
plt.figure(figsize=(10, 6))
for name, model in models:
    # evaluate model with cross validation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1234)
    scores = cross_validate(model, X, y, scoring=["accuracy", "recall"], cv=cv, n_jobs=-1)
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
    data_holdout.append([f"{name}\nAccuracy", accuracy_score(y_test, y_pred)])
    data_holdout.append([f"{name}\nRecall", recall_score(y_test, y_pred)])
    for fold in range(len(scores["test_accuracy"])):
        for score in ["accuracy", "recall"]:
            data_cv.append([f"{name}\n{score.capitalize()}", name, scores[f"test_{score}"][fold]])
df_cv = pd.DataFrame(data_cv, columns=["model_score", "model", "score"])
df_holdout = pd.DataFrame(data_holdout, columns=["model_score", "score"])
df_runtime = pd.DataFrame(data_runtime, columns=["model", "train_time", "pred_time"])
meanlineprops = dict(linestyle='--', linewidth=1.5, color='white')
my_pal = {model_score: "C0" if "LinearSVC" in model_score else "C1" if "Ridge" in model_score else "C2"
          for model_score in df_cv["model_score"].unique()}
sns.set(style="darkgrid")
sns.color_palette("pastel")
ax = sns.boxplot(x="model_score", y="score", data=df_cv, palette=my_pal, showmeans=True,
                 meanline=True, meanprops=meanlineprops)
plt_holdout, = plt.plot(df_holdout["model_score"], df_holdout["score"], "x", color="red",
                        markersize=10, markeredgewidth=2, label="Holdout Result")
mean_line_dummy, = plt.plot([], [], '--', linewidth=1.5, color='white', label='Mean')
plt.legend(handles=[plt_holdout, mean_line_dummy], loc="lower right")
plt.xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(Config.PLOTS_DIR, "model_comparison.pdf"))

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