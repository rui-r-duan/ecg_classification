from common_load import load_data
import numpy as np

X, y = load_data()

# splitting the dataset to training and validation datasets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# -------- Predicting with XGBClassifier
import xgboost
xgb = xgboost.XGBClassifier(objective="multi:softprob", nthread=-1)
xgb.fit(X_train, y_train)
y_train_xgb = xgb.predict(X_train)
y_pred_xgb = xgb.predict(X_test)
print('XGB Train Score:', np.mean(y_train == y_train_xgb))
print('XGB Test Score:', np.mean(y_test == y_pred_xgb))
print('XGB Train Score: {:.2f}'.format(xgb.score(X_train, y_train))) # R^2 score: mean accuracy
print('XGB Test Score: {:.2f}'.format(xgb.score(X_test, y_test)))
# 10-fold cross validation for XGB
from sklearn.model_selection import cross_val_score
scores = cross_val_score(xgb, X, y, cv=10)
print("XGB Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# -------- Predicting with Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
y_train_gbrt = gbrt.predict(X_train)
y_pred_gbrt = gbrt.predict(X_test)
print('GBRT Train Score: {:.2f}'.format(gbrt.score(X_train, y_train))) # R^2 score: mean accuracy
print('GBRT Test Score: {:.2f}'.format(gbrt.score(X_test, y_test)))
# 10-fold cross validation for GBRT
from sklearn.model_selection import cross_val_score
scores = cross_val_score(gbrt, X, y, cv=10)
print("GBRT Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# -------- Predicting with Random Forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_jobs=-1, random_state=0)
forest.fit(X_train, y_train)
y_train_forest = forest.predict(X_train)
y_pred_forest = forest.predict(X_test)
print('Random Forest Train Score:', np.mean(y_train == y_train_forest))
print('Random Forest Test Score:', np.mean(y_test == y_pred_forest))

# -------- Predicting with Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=0.1) # C controls the strength of regularization, smaller value, stronger regularization
lr.fit(X_train, y_train)
y_train_lr = lr.predict(X_train)
y_pred_lr = lr.predict(X_test)
print('Logistic Regression Train Score:', np.mean(y_train == y_train_lr))
print('Logistic Regression Test Score:', np.mean(y_test == y_pred_lr))

# -------- Predicting with Ensemble Voting based on the above classifiers
from sklearn.ensemble import VotingClassifier

eclf = VotingClassifier(estimators=[('xgboost', xgb), ('gbrt', gbrt), ('forest', forest),
                                    ('logistic regression', lr)],
                        voting='soft',
                        weights=None) #[2, 5, 2, 1]) # None: uses uniform weights
eclf = eclf.fit(X_train, y_train)
y_train_ensemble = eclf.predict(X_train)
y_pred_ensemble = eclf.predict(X_test)
print('Ensemble Voting Train Score:', np.mean(y_train == y_train_ensemble))
print('Ensemble Voting Test Score:', np.mean(y_test == y_pred_ensemble))
