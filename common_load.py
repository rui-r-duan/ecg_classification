from dataset import read_dataset, impute, normalizing


def load_data():
    # read the data
    X, y = read_dataset()

    # impute the missing data in the dataset
    X = impute(X)

    # normalizing the dataset
    X = normalizing(X)
    return X, y
