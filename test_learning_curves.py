from common_load import load_data
import xgboost
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from plot_learning_curve import plot_learning_curve
import matplotlib.pyplot as plt


X, y = load_data()

# Learning Curves for LogisticRegression Tuning
lr_a = LogisticRegression() # C=1.0
lr_b = LogisticRegression(C=0.1)
lr_c = LogisticRegression(C=0.03)

plt.figure()
plot_learning_curve(lr_a, X, y, 'C=1.0')
plot_learning_curve(lr_b, X, y, 'C=0.1')
plot_learning_curve(lr_c, X, y, 'C=0.03')
plt.legend(loc=(0, 1.00), ncol=2, fontsize=11)
plt.savefig('LogisticRegression_Tuning' + '.png', format='png')

# Learning Curves for all the tuned classifiers
xgb = xgboost.XGBClassifier(objective="multi:softprob", nthread=-1)
gbrt = GradientBoostingClassifier(random_state=0)
forest = RandomForestClassifier(n_jobs=-1, random_state=0)

plt.figure()
plot_learning_curve(xgb, X, y, 'xgb')
plot_learning_curve(gbrt, X, y, 'gbrt')
plot_learning_curve(forest, X, y, 'forest')
plot_learning_curve(lr_c, X, y, 'LR')
plt.legend(loc=(0, 1.00), ncol=2, fontsize=11)
plt.savefig('learning_curves_all' + '.png', format='png')