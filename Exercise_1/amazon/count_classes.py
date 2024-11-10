from utils import Config
import os
import pandas as pd

df = pd.read_csv(os.path.join(Config.DATA_DIR, "amazon_review_ID.shuf.lrn.csv"))
df_test = pd.read_csv(os.path.join(Config.DATA_DIR, "amazon_review_ID.shuf.tes.csv"))


# Count the classes in the training dataset
class_counts_train = df['Class'].nunique()
print("Class counts in training dataset:")
print(class_counts_train)

