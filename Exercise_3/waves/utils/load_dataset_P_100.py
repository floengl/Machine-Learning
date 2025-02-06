import os
import pandas as pd
from .config import Config


def load_dataset_P_100():
    df = pd.read_csv(os.path.join(Config.DATA_DIR, "WEC_Perth_100.csv"))
    target = df["Total_Power"]
    columns_to_drop = [f"Power{i}" for i in range(1, 101)]
    X = df.drop(columns_to_drop+["qW","Total_Power"], axis=1)
    return X, target

