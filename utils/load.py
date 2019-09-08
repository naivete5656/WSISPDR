import random
from pathlib import Path
from skimage.feature import peak_local_max
import numpy as np
import torch
import cv2
from scipy.ndimage.interpolation import rotate


def to_cropped_imgs(ids, dir_img):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        try:
            yield cv2.imread(str(dir_img / id.name))[:, :, :1].astype(np.float32)
        except TypeError:
            print(str(dir_img / id.name))


def to_cropped_imgs2(dir_imgs):
    """From a list of tuples, returns the correct cropped img"""
    for dir_img in dir_imgs:
        yield cv2.imread(str(dir_img))[:, :, :1].astype(np.float32)


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])


def normalize(x):
    return x / 255


def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b


def local_maxima(img, threshold=100, dist=2):
    assert len(img.shape) == 2
    data = np.zeros((0, 2))
    x = peak_local_max(img, threshold_abs=threshold, min_distance=dist)
    peak_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for j in range(x.shape[0]):
        peak_img[x[j, 0], x[j, 1]] = 255
    labels, _, _, center = cv2.connectedComponentsWithStats(peak_img)
    for j in range(1, labels):
        data = np.append(data, [[center[j, 0], center[j, 1]]], axis=0).astype(int)
    return data


class CellImageLoad(object):
    def __init__(self, ori_path, gt_path, crop_size=(256, 256)):
        self.ori_paths = ori_path
        self.gt_paths = gt_path
        self.crop_size = crop_size

    def __len__(self):
        return len(self.ori_paths) - 1

    def random_crop_param(self, shape):
        h, w = shape
        top = np.random.randint(0, h - self.crop_size[0])
        left = np.random.randint(0, w - self.crop_size[1])
        bottom = top + self.crop_size[0]
        right = left + self.crop_size[1]
        return top, bottom, left, right

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        img = cv2.imread(str(img_name), 0)[:880]
        img = img / img.max()

        gt_name = self.gt_paths[data_id]
        gt = cv2.imread(str(gt_name), 0)
        gt = gt / gt.max()

        # data augumentation
        top, bottom, left, right = self.random_crop_param(img.shape)

        img = img[top:bottom, left:right]
        gt = gt[top:bottom, left:right]

        rand_value = np.random.randint(0, 4)
        img = rotate(img, 90 * rand_value, mode="nearest")
        gt = rotate(gt, 90 * rand_value)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        datas = {"image": img.unsqueeze(0), "gt": gt.unsqueeze(0)}

        return datas


def get_feature_map(ids, dir_after):
    for id in ids:
        yield np.load(dir_after / Path(id.name[:-4] + ".npy"))[0]
