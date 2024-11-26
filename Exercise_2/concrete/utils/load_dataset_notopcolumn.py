import os
import pandas as pd
from .config import Config

def load_dataset_notopcolumn():
    df = pd.read_excel(os.path.join(Config.DATA_DIR, "Concrete_Data.xls"))
    # Drop the top row if it contains headers or metadata
    df.iloc[1: , :]
    target = df["compressive_strength"]
    X = df.drop("compressive_strength", axis=1)
    return X, target