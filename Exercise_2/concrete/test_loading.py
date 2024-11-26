from utils import load_dataset, load_dataset_notopcolumn
from sklearn.model_selection import train_test_split


X,y=load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(X.sample)
print(y.sample)