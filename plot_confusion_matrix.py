import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=matplotlib.cm.get_cmap("jet")):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def gen_confusion_matrix_figure(conf_arr, filename):
    font = {'size': 11}
    plt.rc('font', **font)

    norm_conf = []
    for i in conf_arr:
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            if a == 0:
                tmp_arr.append(0)
            else:
                tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=matplotlib.cm.get_cmap("jet"),
                    interpolation='nearest')

    width = len(conf_arr)
    height = len(conf_arr[0])

    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    fig.colorbar(res)
    axis_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                   '11', '12', '13', '14', '15', '16']
    plt.xticks(range(width), axis_labels)
    plt.yticks(range(height), axis_labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(filename + '.eps', format='eps')
    plt.savefig(filename + '.png', format='png')
