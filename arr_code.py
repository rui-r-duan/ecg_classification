import numpy as np
import pandas as pd

# remove question mark (missing values) in the raw file
with open('arrhythmia.data.orig') as inputfile:
    with open('arrhythmia.data', 'w') as outputfile:
        for line in inputfile:
            outputfile.write(line.replace('?', ''))

# read the data
df = pd.read_csv('arrhythmia.data', header=None)
y = df.iloc[:, -1]  # The last column is the ground-truth label vector
X = df.iloc[:, :-1]  # The first to second-last columns are the features

# impute the missing data in the dataset
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X = imp.fit_transform(X)

# normalizing the dataset
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)

# splitting the dataset to training and validation datasets
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

# -------- Predicting with XGBClassifier
# Load the classifier
import xgboost as xgb

model = xgb.XGBClassifier(objective="multi:softprob", nthread=-1)

# Fit the classifier to the training data
model.fit(X_train, y_train)

# Predicting the results
y_train_xgb = model.predict(X_train)
y_pred_xgb = model.predict(X_val)
print('XGB Train Score:', np.mean(y_train == y_train_xgb))
print('XGB Val Score:', np.mean(y_val == y_pred_xgb))

'''
Outputs:
XGB Train Score: 1.0
XGB Val Score: 0.764705882353
'''

# -------- Predicting with Random Forest
# Load the classifier
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_jobs=-1, random_state=0)

# Fit the classifier to the training data.
forest.fit(X_train, y_train)

# Predicting the results
y_train_forest = forest.predict(X_train)
y_pred_forest = forest.predict(X_val)
print('Random Forest Train Score:', np.mean(y_train == y_train_forest))
print('Random Forest Val Score:', np.mean(y_val == y_pred_forest))

'''
Outputs:
Random Forest Train Score: 0.981012658228
Random Forest Val Score: 0.713235294118
'''

# -------- Predicting with Logistic Regression
# Load the classifier
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

# Fit the classifier to the training data
lr.fit(X_train, y_train)

# Predicting the results
y_train_lr = lr.predict(X_train)
y_pred_lr = lr.predict(X_val)
print('Logistic Regression Train Score:', np.mean(y_train == y_train_lr))
print('Logistic Regression Val Score:', np.mean(y_val == y_pred_lr))

'''
Outputs:
Logistic Regression Train Score: 0.990506329114
Logistic Regression Val Score: 0.705882352941
'''
