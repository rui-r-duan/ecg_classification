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
import xgboost
xgb = xgboost.XGBClassifier(objective="multi:softprob", nthread=-1)
xgb.fit(X_train, y_train)
y_train_xgb = xgb.predict(X_train)
y_pred_xgb = xgb.predict(X_val)
print('XGB Train Score:', np.mean(y_train == y_train_xgb))
print('XGB Val Score:', np.mean(y_val == y_pred_xgb))
print('XGB Train Score: {:.2f}'.format(xgb.score(X_train, y_train))) # R^2 score: mean accuracy
print('XGB Val Score: {:.2f}'.format(xgb.score(X_val, y_val)))
# 10-fold cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(xgb, X, y, cv=10)
print("XGB Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

'''
Outputs:
XGB Train Score: 1.0
XGB Val Score: 0.75
XGB Accuracy: 0.74 (+/- 0.10)
'''

# -------- Predicting with Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
y_train_gbrt = gbrt.predict(X_train)
y_pred_gbrt = gbrt.predict(X_val)
print('GBRT Train Score:', np.mean(y_train == y_train_gbrt))
print('GBRT Val Score:', np.mean(y_val == y_pred_gbrt))

'''
Outputs:
GBRT Train Score: 1.0
GBRT Val Score: 0.772058823529
'''

# -------- Predicting with Random Forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_jobs=-1, random_state=0)
forest.fit(X_train, y_train)
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
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=0.1) # C controls the strength of regularization, smaller value, stronger regularization
lr.fit(X_train, y_train)
y_train_lr = lr.predict(X_train)
y_pred_lr = lr.predict(X_val)
print('Logistic Regression Train Score:', np.mean(y_train == y_train_lr))
print('Logistic Regression Val Score:', np.mean(y_val == y_pred_lr))

'''
Outputs:
Logistic Regression Train Score: 0.990506329114
Logistic Regression Val Score: 0.705882352941
'''

# -------- Predicting with Ensemble Voting based on the above classifiers
from sklearn.ensemble import VotingClassifier

eclf = VotingClassifier(estimators=[('xgboost', xgb), ('gbrt', gbrt), ('forest', forest),
                                    ('logistic regression', lr)],
                        voting='soft',
                        weights=None)#[2, 5, 2, 1]) # None: uses uniform weights
eclf = eclf.fit(X_train, y_train)
y_train_ensemble = eclf.predict(X_train)
y_pred_ensemble = eclf.predict(X_val)
print('Ensemble Voting Train Score:', np.mean(y_train == y_train_ensemble))
print('Ensemble Voting Val Score:', np.mean(y_val == y_pred_ensemble))

'''
Outputs:
Ensemble Voting Train Score: 1.0
Ensemble Voting Val Score: 0.786764705882
'''
