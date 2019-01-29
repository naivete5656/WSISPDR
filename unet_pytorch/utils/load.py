#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import random
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from .utils import normalize, hwc_to_chw


def get_feature_map(ids, dir_after):
    for id in ids:
        yield np.load(dir_after / Path(id.name[:-4] + '.npy'))[0]


def to_cropped_imgs(ids, dir_img):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        yield cv2.imread(str(dir_img / id.name))[:, :, :1].astype(np.float32)


def get_imgs_and_masks(dir_img, dir_mask):
    """Return all the couples (img, mask)"""
    paths = list(dir_img.glob('*.tif'))
    random.shuffle(list(paths))
    imgs = to_cropped_imgs(paths, dir_img)
    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)
    masks = to_cropped_imgs(paths, dir_mask)
    masks_switched = map(hwc_to_chw, masks)
    masks_normalized = map(normalize, masks_switched)
    return zip(imgs_normalized, masks_normalized)


def get_imgs_cascade(dir_img, dir_before, dir_after):
    """Return all the couples (img, mask)"""
    paths = list(dir_img.glob('*.tif'))
    random.shuffle(list(paths))
    imgs = to_cropped_imgs(paths, dir_img)
    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(paths, dir_before)
    masks_switched = map(hwc_to_chw, masks)
    before_normalized = map(normalize, masks_switched)

    masks = to_cropped_imgs(paths, dir_before)
    masks_switched = map(hwc_to_chw, masks)
    after_normalized = map(normalize, masks_switched)
    return zip(imgs_normalized, before_normalized, after_normalized)


def get_imgs_internal(dir_img, dir_before, dir_after):
    """Return all the couples (img, mask)"""
    paths = list(dir_img.glob('*.tif'))
    random.shuffle(list(paths))
    imgs = to_cropped_imgs(paths, dir_img)
    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(paths, dir_before)
    masks_switched = map(hwc_to_chw, masks)
    before_normalized = map(normalize, masks_switched)

    feature = get_feature_map(paths, dir_after)
    return zip(imgs_normalized, before_normalized, feature)


def get_imgs_multi(ori, ex1, ex2, ex3):
    paths = list(ori.glob('*.tif'))
    random.shuffle(list(paths))

    imgs = to_cropped_imgs(paths, ori)
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    imgs = to_cropped_imgs(paths, ex1)
    imgs_switched = map(hwc_to_chw, imgs)
    ex1_normalized = map(normalize, imgs_switched)

    imgs = to_cropped_imgs(paths, ex2)
    imgs_switched = map(hwc_to_chw, imgs)
    ex2_normalized = map(normalize, imgs_switched)

    imgs = to_cropped_imgs(paths, ex3)
    imgs_switched = map(hwc_to_chw, imgs)
    ex3_normalized = map(normalize, imgs_switched)

    return zip(imgs_normalized, ex1_normalized, ex2_normalized, ex3_normalized)


# 画像を分割今は576に分割
def split_image(img):
    img1 = img[0:576, 0:576]
    img2 = img[464:1040, 0:576]
    img3 = img[0:576, 456:1032]
    img4 = img[464:1040, 456:1032]
    img5 = img[0:576, 816:1392]
    img6 = img[464:1040, 816:1392]
    x = np.array([img1, img2, img3, img4, img5, img6])
    return x


# 　画像を分割して
def update_data(u_d, data, i, ch):
    data = split_image(data)
    for j in range(6):
        u_d[i * 6 + j, :, :, ch] = data[j]
    return u_d


def data_load_split(path, height, width, norm_value):
    paths = sorted((list(path.glob('*.tif'))))
    x = np.zeros((len(paths) * 6, height, width, 1), dtype='float32')
    for i, path in enumerate(paths):
        data = np.array(Image.open(path))
        x = update_data(x, data, i, 0)
    x = x / norm_value
    print('x_gen')
    print(x.shape)
    return x


def cut_image(plot_size='6', sequence=18, size=320, override=100, norm_value=255):
    import cv2
    # paths = sorted(Path('../image/frame2/%s' % plot_size).glob('*.tif'))
    paths = sorted(Path('../image/challenge2').glob('*.tif'))
    i = 0
    # savepath = Path('../image/gt_%s/%s' % (sequence, plot_size))
    # savepath = Path('../image/ori_%s' % (sequence))
    # savepath = Path('../image/challenge-train/frame2/%s' % plot_size)
    savepath = Path('../image/challenge-train/frame2/ori')
    savepath.mkdir(parents=True, exist_ok=True)
    for path in paths:
        img = np.array(Image.open(path)).astype(np.float32)
        img = (img / norm_value * 255).astype(np.uint8)
        for x in range(0, img.shape[0] - size, size - override):
            for y in range(0, img.shape[1] - size, size - override):
                cv2.imwrite(str(savepath / Path('%05d.tif' % i)), img[x:x + size, y:y + size])
                i += 1
            cv2.imwrite(str(savepath / Path('%05d.tif' % i)), img[x: x + size, img.shape[1] - size: img.shape[1]])
            i += 1
        for y in range(0, img.shape[1] - size, size - override):
            cv2.imwrite(str(savepath / Path('%05d.tif' % i)), img[img.shape[0] - size:img.shape[0], y:y + size])
            i += 1
        cv2.imwrite(str(savepath / Path('%05d.tif' % i)),
                    img[img.shape[0] - size:img.shape[0], img.shape[1] - size: img.shape[1]])
        i += 1
    print(i)


if __name__ == '__main__':
    cut_image(plot_size='3')
    cut_image(plot_size='9')
    cut_image(plot_size='12')
