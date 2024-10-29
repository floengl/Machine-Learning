import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .config import Config


def load_training_dataset(return_label_encoder=False):
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
             'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    data = pd.read_csv(os.path.join(Config.DATA_DIR, "adult.data"), header=None, na_values='?', skipinitialspace=True,
                       names=names)
    test = pd.read_csv(os.path.join(Config.DATA_DIR, "adult.test"), header=None, skiprows=1, na_values='?', names=names,
                       skipinitialspace=True)
    test["income"] = test["income"].str.rstrip(".")  # remove trailing dot from income column
    df = pd.concat([data, test]).reset_index(drop=True)  # combine data and test
    target = df["income"]
    X = df.drop(["income", "education"], axis=1)  # education is already represented by education-num
    le = LabelEncoder()
    y = le.fit_transform(target)
    if return_label_encoder:
        return X, y, le
    else:
        return X, y


# divide features into categorical and numerical
categorical = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
numeric = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
