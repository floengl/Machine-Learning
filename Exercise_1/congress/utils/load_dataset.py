import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .config import Config


def load_training_dataset(return_label_encoder=False):
    df = pd.read_csv(os.path.join(Config.DATA_DIR, "CongressionalVotingID.shuf.lrn.csv"), na_values="unknown")
    target = df["class"]
    X = df.drop("class", axis=1)
    le = LabelEncoder()
    y = le.fit_transform(target)
    if return_label_encoder:
        return X, y, le
    else:
        return X, y


def load_test_dataset():
    df = pd.read_csv(os.path.join(Config.DATA_DIR, "CongressionalVotingID.shuf.tes.csv"), na_values="unknown")
    return df
