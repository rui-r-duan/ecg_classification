from common_load import load_data
import xgboost
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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

classifier_list = [xgb, gbrt, forest, lr, eclf]

for clf in classifier_list:
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    y_train_pred = clf.fit(X_train, y_train).predict(X_train)

    # Compute confusion matrix
    cnf_test = confusion_matrix(y_test, y_pred, labels=range(1, 17))
    cnf_train = confusion_matrix(y_train, y_train_pred, labels=range(1, 17))

    # Plot confusion matrix
    gen_confusion_matrix_figure(cnf_test,
                                clf.__class__.__name__ + "_test",
                                is_save_to_file=True)
    gen_confusion_matrix_figure(cnf_train,
                                clf.__class__.__name__ + "_train",
                                is_save_to_file=True)
