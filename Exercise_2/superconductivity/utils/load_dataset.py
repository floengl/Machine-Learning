import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .config import Config


def load_training_dataset():
    df = pd.read_csv(os.path.join(Config.DATA_DIR, "train.csv"))
    target = df["critical_temp"]
    X = df.drop("critical_temp", axis=1)
    return X, target

