from matplotlib_venn import venn3
import matplotlib.pyplot as plt
import numpy as np


def ven3_diagram():
    data = np.loadtxt('each_expert_tp.csv', delimiter=',', skiprows=1)
    detect_total = data[(data[:, 1] == 1) | (data[:, 2] == 1) | (data[:, 3] == 1)].shape[0]
    detect_all = data[(data[:, 1] == 1) & (data[:, 2] == 1) & (data[:, 3] == 1)].shape[0] / 17713
    detect_12 = data[(data[:, 1] == 1) & (data[:, 2] == 1) & (data[:, 3] == 0)].shape[0] / 17713
    detect_23 = data[(data[:, 1] == 0) & (data[:, 2] == 1) & (data[:, 3] == 1)].shape[0] / 17713
    detect_13 = data[(data[:, 1] == 1) & (data[:, 2] == 0) & (data[:, 3] == 1)].shape[0] / 17713
    detect_only_1 = data[(data[:, 1] == 1) & (data[:, 2] == 0) & (data[:, 3] == 0)].shape[0] / 17713
    detect_only_2 = data[(data[:, 1] == 0) & (data[:, 2] == 1) & (data[:, 3] == 0)].shape[0] / 17713
    detect_only_3 = data[(data[:, 1] == 0) & (data[:, 2] == 0) & (data[:, 3] == 1)].shape[0] / 17713
    detect_1 = data[data[:, 1] == 1].shape[0] / 17713
    detect_2 = data[data[:, 2] == 1].shape[0] / 17713
    detect_3 = data[data[:, 3] == 1].shape[0] / 17713

    total = detect_total / 17713
    plt.title('%f' % total)
    venn3(subsets={'100': detect_only_1, '010': detect_only_2, '001': detect_only_3,
                   '110': detect_12, '101': detect_13, '011': detect_23,
                   '111': detect_all}, set_labels=('3:%f' % detect_1, '6:%f' % detect_2, '9:%f' % detect_3))
    plt.show()


if __name__ == '__main__':
    ven3_diagram()
