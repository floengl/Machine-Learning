from utils import load_training_dataset, setup_logging 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Set up logging
logger = setup_logging("ev_rf")

# load dataset
X, y = load_training_dataset()

# define preprocessing pipeline
preprocessor = Pipeline([
    ("remove_ID", ColumnTransformer([("remove_ID", "drop", "ID")], remainder="passthrough")),
    ("pow_tra", PowerTransformer())
])

# define estimator
estimator = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=1234,n_estimators=2000,n_jobs=-1,max_depth=45,min_samples_split=2,min_samples_leaf=1,max_features="sqrt"))
])

f1= []
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1234)
scores = cross_validate(estimator, X, y, scoring=["f1_macro"], cv=cv, n_jobs=-1)

f1.append((np.mean(scores['test_f1_macro']), np.std(scores['test_f1_macro'])))


logger.info(f"\nF1 random forest:")
logger.info(pd.DataFrame(f1, columns=['mean', 'std']).sort_values(by='mean', ascending=False))
