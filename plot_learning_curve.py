import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve, KFold


def plot_learning_curve(est, X, y, estimator_name):
    training_set_size, train_scores, test_scores = learning_curve(
        est, X, y, train_sizes=np.linspace(.1, 1, 5), cv=KFold(5, shuffle=True, random_state=1))
    line = plt.plot(training_set_size, train_scores.mean(axis=1), '--',
                    label="training " + estimator_name)
    plt.plot(training_set_size, test_scores.mean(axis=1), '-',
             label="test " + estimator_name, c=line[0].get_color())
    plt.xlabel('Training set size')
    plt.ylabel('Score (R^2)')
    plt.ylim(0, 1.1)
