import random
from pathlib import Path
from skimage.feature import peak_local_max
import numpy as np
import cv2
import collections
import torch


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


# def get_feature_map(ids, dir_after):
#     for id in ids:
#         yield np.load(dir_after / Path(id.name[:-4] + ".npy"))[0]
#
#
# def to_cropped_imgs(ids, dir_img):
#     """From a list of tuples, returns the correct cropped img"""
#     for id in ids:
#         yield cv2.imread(str(dir_img / id.name))[:, :, :1].astype(np.float32)
#
#
# def get_imgs_and_masks_boundaries(dir_img, dir_mask, dir_boundary):
#     """Return all the couples (img, mask)"""
#     paths = list(dir_img.glob("*.tif"))
#     random.shuffle(list(paths))
#
#     imgs = to_cropped_imgs(paths, dir_img)
#     imgs_switched = map(hwc_to_chw, imgs)
#     imgs_normalized = map(normalize, imgs_switched)
#
#     masks = to_cropped_imgs(paths, dir_mask)
#     masks_switched = map(hwc_to_chw, masks)
#     # masks_normalized = map(normalize, masks_switched)
#
#     boundaries = to_cropped_imgs(paths, dir_boundary)
#     boundaries_switched = map(hwc_to_chw, boundaries)
#     # boundaries_normalized = map(normalize, boundaries_switched)
#
#     return zip(imgs_normalized, masks_switched, boundaries_switched)
#
#
# def hwc_to_chw(img):
#     return np.transpose(img, axes=[2, 0, 1])
#
#
# def normalize(x):
#     return x / 255


# def batch(iterable, batch_size):
#     """Yields lists by batch"""
#     b = []
#     for i, t in enumerate(iterable):
#         b.append(t)
#         if (i + 1) % batch_size == 0:
#             yield b
#             b = []
#
#     if len(b) > 0:
#         yield b
#
#
# def detection_with_network(img, net, gpu=True):
#     img = (img.astype(np.float32) / 255).reshape((1, img.shape[0], img.shape[1]))
#     with torch.no_grad():
#         img = torch.from_numpy(img).unsqueeze(0)
#         if gpu:
#             img = img.cuda()
#         mask_pred = net(img)
#     pre_img = mask_pred.detach().cpu().numpy()[0, 0]
#     pre_img = (pre_img * 255).astype(np.uint8)
#     peaks = local_maxima(pre_img, 100, 10)
#     return pre_img, peaks
