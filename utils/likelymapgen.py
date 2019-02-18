"""
Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields
の(7)式を単純な方法で実現
・xmlファイルにある重心座標から尤度マップを生成
・shapeは(1040,1392)
・めちゃ時間かかる(xml操作の効率化)
"""

import xml.etree.ElementTree as ET
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def gaus_filter(img, kernel_size, sigma):
    pad_size = int(kernel_size - 1 / 2)
    img_t = np.pad(img, (pad_size, pad_size), 'constant')  # zero padding(これしないと正規化後、画像端付近の尤度だけ明るくなる)
    img_t = cv2.GaussianBlur(img_t, ksize=(kernel_size, kernel_size), sigmaX=sigma)  # filter gaussian(適宜パラメータ調整)
    img_t = img_t[pad_size:-pad_size, pad_size:-pad_size]  # remove padding
    return img_t


def like_map_gen(save_path, plot_size):
    save_path = Path('/home/kazuya/weakly_supervised_instance_segmentation/images/sequence/sequ9/%d' % plot_size)
    save_path.mkdir(parents=True, exist_ok=True)
    # load xml file
    tree = ET.parse('./090303_exp1_F0009_GT_full.xml')
    # tree = ET.parse('./unet/sequence18.xml')
    root = tree.getroot()

    # number of cell
    num_cell = len(root[0].findall("Cell"))

    black = np.zeros((1040, 1392))

    # 1013 - number of frame
    for i in range(0, 780):
        # likelihood map of one input
        result = black.copy()
        result_big = black.copy()

        for j in range(num_cell):
            index_num = (".//Frame[@index=\'" + str(i + 1) + "\']")
            index_check = len(root[0][j][0].findall(index_num))
            if index_check != 0:
                data = root[0][j][0].find(index_num)[0][0].text[1:-1]  # coordinate of center of gravity
                data = data.split(' ')
                x = int(float(data[0]))
                y = int(float(data[1]))
                img_t = black.copy()  # likelihood map of one cell
                img_t[y][x] = 255  # plot a white dot
                img_t = gaus_filter(img_t, 301, plot_size)
                result = np.maximum(result, img_t)  # compare result with gaussian_img
        #  normalization
        result = 255 * result / result.max()
        result = result.astype('uint8')
        cv2.imwrite(str(save_path / Path('%05d.tif' % i)), result)
        print(i + 1)
    print('finish')


def like_map_gen_handmade(plot_size, height=1040, width=1392):
    save_path = Path('/home/kazuya/weakly_supervised_instance_segmentation/images/sequence/sequ17/%d' % plot_size)
    save_path.mkdir(parents=True, exist_ok=True)
    # load xml file
    # tree = ET.parse('./unet/090303_exp1_F0009_GT_full.xml')
    # tree = ET.parse('challenge01.xml')
    tree = ET.parse('/home/kazuya/ctk/sequence17.xml')
    root = tree.getroot()
    annotations = []
    for i in root.findall(".//s"):
        annotations.append([int(float(i.get('i'))), int(float(i.get('x'))), int(float(i.get('y')))])

    # number of cell
    annotations = np.array(annotations)
    # 1013 - number of frame
    for i in range(50, 100):
        # likelihood map of one input
        result = np.zeros((height, width))
        frame_per_annotations = annotations[annotations[:, 0] == i]
        for annotation in frame_per_annotations:
            img_t = np.zeros((height, width))  # likelihood map of one cell
            img_t[annotation[2]][annotation[1]] = 255  # plot a white dot
            img_t = gaus_filter(img_t, 301, plot_size)
            result = np.maximum(result, img_t)  # compare result with gaussian_img
        #  normalization
        result = 255 * result / result.max()
        result = result.astype('uint8')
        cv2.imwrite(str(save_path / Path('%05d.tif' % i)), result)
        print(i + 1)
    print('finish')


def challenge_gt_gen(plot_size):
    save_path = Path('../image/challenge/train_gt/%d' % plot_size)
    save_path.mkdir(parents=True, exist_ok=True)
    path = Path('../image/challenge/train_gt/TRA')
    paths = sorted(list(path.glob('*.tif')))
    for i, path in enumerate(paths):
        img = cv2.imread(str(path), -1)
        res = np.zeros((img.shape[0], img.shape[1]))
        for j in range(1, img.max() + 1):
            x, y = np.where(img == j)
            img_t = np.zeros((img.shape[0], img.shape[1]))
            try:
                x = int(x.sum() / x.shape[0])
                y = int(y.sum() / y.shape[0])
                img_t[x, y] = 255
                img_t = gaus_filter(img_t, 101, plot_size)
                res = np.maximum(res, img_t)  # compare result with gaussian_img
            except ValueError:
                pass
        res = 255 * res / res.max()
        res = res.astype('uint8')
        cv2.imwrite(str(save_path / Path('%05d.tif' % i)), res)
        print(i)
    print('finish')


def cut_image(plot_size='6', sequence=18, size=320, override=100, norm_value=255):
    import cv2
    # paths = sorted(Path(f'../image/challenge/gt1/{plot_size}').glob('*.tif'))
    paths = sorted(Path(f'../images/pre').glob('*.tif'))
    # paths = sorted(Path('../image/challenge/challenge1').glob('*.tif'))
    i = 0
    # savepath = Path('../image/gt_%s/%s' % (sequence, plot_size))
    # savepath = Path('../image/ori_%s' % (sequence))
    # savepath = Path(f'../image/challenge/cut/frame1/{plot_size}')
    savepath = Path(f'../images/phase_off')
    # savepath = Path('../image/challenge/challenge/frame1/ori')
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
    # like_map_gen(Path, 20)
    # challenge_gt_gen(3)
    # like_map_gen_handmade(3, height=520, width=696)
    # like_map_gen_handmade(6, height=520, width=696)
    # like_map_gen_handmade(9, height=520, width=696)
    # like_map_gen_handmade(12, height=520, width=696)
    # like_map_gen_handmade(20, height=520, width=696)
    # like_map_gen_handmade(3)
    # like_map_gen_handmade(6)
    # like_map_gen_handmade(9)
    like_map_gen_handmade(20)

    # load_and_concatenate()
    # cut_image(plot_size='3')
    # cut_image(plot_size='6')
    # cut_image(plot_size='9')
    # cut_image(plot_size='12')
    # cut_image(plot_size='20')
