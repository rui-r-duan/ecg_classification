from common_load import load_data
import xgboost
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from plot_learning_curve import plot_learning_curve
import matplotlib.pyplot as plt


X, y = load_data()

#xgb = xgboost.XGBClassifier(objective="multi:softprob", nthread=-1)
#gbrt = GradientBoostingClassifier(random_state=0)
#forest = RandomForestClassifier(n_jobs=-1, random_state=0)
lr_a = LogisticRegression() # C=1.0
lr_b = LogisticRegression(C=0.1)
lr_c = LogisticRegression(C=0.03)

plot_learning_curve(lr_a, X, y)
plot_learning_curve(lr_b, X, y)
plot_learning_curve(lr_c, X, y)
plt.legend(loc=(0, 1.00), ncol=3, fontsize=11)
plt.show()