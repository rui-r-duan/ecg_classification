import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

def gen_confusion_matrix_figure(conf_arr, figure_name,
                                is_save_to_file=False):
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

    if is_save_to_file:
        plt.savefig(figure_name + '.png', format='png')
    else:
        fig.suptitle(figure_name, fontsize=12, fontweight='bold')
        plt.show()
