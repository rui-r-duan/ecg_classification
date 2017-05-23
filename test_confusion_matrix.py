from common_load import load_data
import xgboost
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# from plot_confusion_matrix import plot_confusion_matrix
from plot_confusion_matrix import gen_confusion_matrix_figure


X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

xgb = xgboost.XGBClassifier(objective="multi:softprob", nthread=-1,
                            reg_alpha=0.7, reg_lambda=0.05, subsample=0.9)
gbrt = GradientBoostingClassifier(random_state=0)
forest = RandomForestClassifier(n_jobs=-1, random_state=0)
lr = LogisticRegression(C=0.03)
eclf = VotingClassifier(estimators=[('xgboost', xgb), ('gbrt', gbrt), ('forest', forest),
                                    ('logistic regression', lr)],
                        voting='soft',
                        weights=None)

y_pred = eclf.fit(X_train, y_train).predict(X_test)
y_train_pred = eclf.fit(X_train, y_train).predict(X_train)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred, labels=range(1, 17))
cnf_train = confusion_matrix(y_train, y_train_pred, labels=range(1, 17))
# np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=range(1,13), normalize=True,
#                       title='Confusion matrix, without normalization')
#
#
# plt.figure()
# plot_confusion_matrix(cnf_train, classes=range(1,14), normalize=True,
#                       title='Confusion matrix, without normalization')
# plt.show()

gen_confusion_matrix_figure(cnf_matrix, "voting")
gen_confusion_matrix_figure(cnf_train, "voting_train")