import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler


def read_dataset():
    df = pd.read_csv('arrhythmia.data', header=None)
    y = df.iloc[:, -1]  # The last column is the ground-truth label vector
    X = df.iloc[:, :-1]  # The first to second-last columns are the features
    return X, y


def impute(X):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X = imp.fit_transform(X)
    return X


def normalizing(X):
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X