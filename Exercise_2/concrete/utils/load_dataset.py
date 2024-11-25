import os
import pandas as pd
from .config import Config


def load_dataset():
    df = pd.read_excel(os.path.join(Config.DATA_DIR, "Concrete_Data.xls"))
    target = df["compressive_strength"]
    X = df.drop("compressive_strength", axis=1)
    return X, target


