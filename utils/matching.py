import numpy as np
import math
import pulp
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max
from pathlib import Path

def optimum(target, pred, dist_threshold):
    """
    :param target:target plots numpy [x,y]
    :param pred: pred plots numpy[x,y]
    :param dist_threshold: distance threshold
    :return: association result
    """
    r = 0.01
    # matrix to calculate
    c = np.zeros((0, pred.shape[0] + target.shape[0]))
    d = np.array([])
    associate_id = np.zeros((0, 2))

    # GT position との距離を算出
    for ii in range(int(target.shape[0])):
        dist = pred[:, 0:2] - np.tile(target[ii, 0:2], (pred.shape[0], 1))
        # 一点との距離を算出
        dist_lis = np.sqrt(np.sum(np.square(dist), axis=1))
        cc = np.where(dist_lis <= dist_threshold)[0]

        # 可能性がある候補に１を立てる
        for j in cc:
            c1 = np.zeros((1, target.shape[0] + pred.shape[0]))
            c1[0, ii] = 1
            c1[0, target.shape[0] + j] = 1
            c = np.append(c, c1, axis=0)
            d = np.append(d, math.exp(-r * dist_lis[j]))
            associate_id = np.append(associate_id, [[ii, j]], axis=0)

    # 最適化問題を最大を求める問題に設定
    prob = pulp.LpProblem("review", pulp.LpMaximize)

    index = list(range(d.shape[0]))  # type:

    x_vars = pulp.LpVariable.dict("x", index, lowBound=0, upBound=1, cat="Integer")

    # 最大化する値
    prob += sum([d[i] * x_vars[i] for i in range(d.shape[0])])

    # 条件の定義
    for j in range(c.shape[1]):
        prob += sum([c[i, j] * x_vars[i] for i in range(d.shape[0])]) <= 1

    # 最適化問題を解く
    prob.solve()

    # 最適化結果を抽出　=0 のデータを削除
    x_list = np.zeros(d.shape[0], dtype=int)
    for jj in range(d.shape[0]):
        x_list[jj] = int(x_vars[jj].value())
    associate_id = np.delete(associate_id, np.where(x_list == 0)[0], axis=0)
    return associate_id


def remove_outside_plot(matrix, associate_id, i, window_size, window_thresh=10):
    """
    delete peak that outside
    :param matrix:target matrix
    :param associate_id:optimizeした結果
    :param i: 0 or 1 0の場合target,1の場合predを対象にする
    :param window_size: window size
    :return: removed outside plots
    """
    # delete edge plot 対応付けされなかった中で端のデータを消去
    index = np.delete(np.arange(matrix.shape[0]), associate_id[:, i])
    a = np.where(
        (matrix[index][:, 0] < window_thresh) | (matrix[index][:, 0] > window_size[1] - window_thresh)
    )[0]
    b = np.where(
        (matrix[index][:, 1] < window_thresh) | (matrix[index][:, 1] > window_size[0] - window_thresh)
    )[0]
    delete_index = np.unique(np.append(a, b, axis=0))
    return (
        np.delete(matrix, index[delete_index], axis=0),
        np.delete(index, delete_index),
    )


def show_res(img, gt, res, no_detected_id, over_detection_id, path=None):
    plt.figure(figsize=(4, 3), dpi=500)
    plt.imshow(img)
    plt.plot(gt[:, 0], gt[:, 1], "y3", label="gt_mapped")
    plt.plot(res[:, 0], res[:, 1], "g4", label="res_mapped")
    plt.plot(
        gt[no_detected_id][:, 0], gt[no_detected_id][:, 1], "b2", label="no_detected"
    )
    plt.plot(
        res[over_detection_id][:, 0],
        res[over_detection_id][:, 1],
        "k1",
        label="over_detection",
    )
    plt.legend(bbox_to_anchor=(0, 1.05), loc="upper left", fontsize=4, ncol=4)
    plt.show()
    # plt.savefig(path)


def local_maxim(img, threshold, dist):
    data = np.zeros((0, 2))
    x = peak_local_max(img, threshold_abs=threshold, min_distance=dist)
    peak_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for j in range(x.shape[0]):
        peak_img[x[j, 0], x[j, 1]] = 255
    labels, _, _, center = cv2.connectedComponentsWithStats(peak_img)
    for j in range(1, labels):
        data = np.append(data, [[center[j, 0], center[j, 1]]], axis=0)
    return data


def target_peaks_gen(img):
    gt_plot = np.zeros((0, 2))
    x, y = np.where(img == 255)
    for j in range(x.shape[0]):
        gt_plot = np.append(gt_plot, [[y[j], x[j]]], axis=0)
    return gt_plot


def gaus_filter(img, kernel_size, sigma):
    pad_size = int(kernel_size - 1 / 2)
    img_t = np.pad(
        img, (pad_size, pad_size), "constant"
    )  # zero padding(これしないと正規化後、画像端付近の尤度だけ明るくなる)
    img_t = cv2.GaussianBlur(
        img_t, ksize=(kernel_size, kernel_size), sigmaX=sigma
    )  # filter gaussian(適宜パラメータ調整)
    img_t = img_t[pad_size:-pad_size, pad_size:-pad_size]  # remove padding
    return img_t



