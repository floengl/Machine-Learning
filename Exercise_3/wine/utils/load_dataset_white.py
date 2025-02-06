import os
import pandas as pd
from .config import Config


def load_dataset_white():
    df = pd.read_csv(os.path.join(Config.DATA_DIR, "winequality-white.csv"),delimiter=";")
    target = df["quality"]
    X = df.drop("quality", axis=1)
    return X, target

