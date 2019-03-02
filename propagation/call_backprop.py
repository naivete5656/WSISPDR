from pathlib import Path
import torch
from PIL import Image
import numpy as np
import cv2
from gradcam import Test3, Test2, Test
from scipy.io import savemat
import matplotlib.pyplot as plt
import sys
from networks import UNet
from utils import local_maxima, gaus_filter


class BackProp(object):
    def __init__(self, input_path, output_path, weight_path, gpu=True, sig=True):
        self.input_path = input_path
        self.output_path = output_path
        output_path.mkdir(parents=True, exist_ok=True)

        self.gpu = gpu
        # network load
        self.net = UNet(n_channels=1, n_classes=1, sig=sig)
        self.net.load_state_dict(
            torch.load(weight_path, map_location={"cuda:3": "cuda:0"})
        )
        self.net.eval()
        if self.gpu:
            self.net.cuda()

        self.shape = None
        self.norm = sig

    def unet_pred(self, img, save_path=None):
        # throw unet
        if self.gpu:
            img = img.cuda()
        pre_img = self.net(img)
        pre_img = pre_img.detach().cpu().numpy()[0, 0]
        if save_path is not None:
            if not self.norm:
                pre_img = (pre_img + 1) / 2
            cv2.imwrite(str(save_path), (pre_img * 255).astype(np.uint8))
        return pre_img

    def coloring(self, gbs):
        # coloring
        r, g, b = np.loadtxt("../utils/color.csv", delimiter=",")
        gbs_coloring = []
        for peak_i, gb in enumerate(gbs):
            gb = gb/gb.max() * 255
            gb = gb.clip(0, 255).astype(np.uint8)
            result = np.ones((self.shape[0], self.shape[1], 3))
            result = gb[..., np.newaxis] * result
            peak_i = peak_i % 20
            result[..., 0][result[..., 0] != 0] = r[peak_i] * gb[gb != 0]
            result[..., 1][result[..., 1] != 0] = g[peak_i] * gb[gb != 0]
            result[..., 2][result[..., 2] != 0] = b[peak_i] * gb[gb != 0]
            gbs_coloring.append(result)
            # cv2.imwrite(
            #     str(self.save_path.joinpath("{:04d}.png".format(peak_i))), result
            # )
        return gbs_coloring

    def main(self):
        for img_i, path in enumerate(self.input_path):
            # self.temp_path = self.text_output.joinpath("{:05d}.txt".format(img_i))
            # with open(self.temp_path, mode="w") as f:
            #     f.write("ID,x,y\n")
            # load image
            img = np.array(Image.open(path))
            self.shape = img.shape
            if self.norm:
                img = (img.astype(np.float32) / 255).reshape(
                    (1, 1, img.shape[0], img.shape[1])
                )
            else:
                img = (img.astype(np.float32) / (255 / 2) - 1).reshape(
                    (1, 1, img.shape[0], img.shape[1])
                )

            img = torch.from_numpy(img)

            pre_img = self.unet_pred(
                img, self.likelymap_savepath.joinpath(f"{img_i:05d}.tif")
            )

            img.requires_grad = True
            gbs = self.calculate(img, pre_img)

            self.save_img(gbs, img_i)


class BackpropAll(BackProp):
    def __init__(self, input_path, output_path, weight_path, gpu=True, radius=1, sig=True):
        super().__init__(input_path, output_path, weight_path, gpu, sig=True)

        self.back_model = Test2(self.net)
        self.likelymap_savepath = output_path.parent.joinpath("likelymap")
        self.likelymap_savepath.mkdir(parents=True, exist_ok=True)

    def calculate(self, img, pre_img):

        mask = np.zeros(pre_img.shape)
        mask[pre_img > 0.1] = 1

        mask_bkg = np.zeros(pre_img.shape)
        mask_bkg[pre_img < 0.1] = 1

        result_fore = self.back_model(img, mask.astype(np.float32)).copy()
        # result_fore = result_fore / result_fore.max()
        return result_fore

    def save_img(self, result, i):
        cv2.imwrite(
            str(self.output_path / Path("%05d.tif" % i)),
            (result / result.max() * 255).clip(0, 255).astype(np.uint8),
        )


class BackPropBackGround(BackProp):
    def __init__(
        self, input_path, output_path, weight_path, gpu=True, radius=1, sig=True
    ):
        super().__init__(input_path, output_path, weight_path, gpu, sig=sig)
        self.output_path_each = None
        self.back_model = Test2(self.net)

    def calculate(self, img, pre_img):
        file_path = self.output_path_each.joinpath("peaks.txt")
        self.save_path = self.output_path_each.joinpath("each_peak")
        self.save_path.mkdir(parents=True, exist_ok=True)

        # peak
        if not self.norm:
            pre_img = (pre_img + 1) / 2
        peaks = local_maxima((pre_img * 255).astype(np.uint8), 125, 2).astype(np.int)
        gauses = []
        try:
            for peak in peaks:
                temp = np.zeros(self.shape)
                temp[peak[1], peak[0]] = 255
                gauses.append(gaus_filter(temp, 401, 12))
            region = np.argmax(gauses, axis=0) + 1
            likely_map = np.max(gauses, axis=0)
            region[pre_img < 0] = 0

            r, g, b = np.loadtxt("../utils/color.csv", delimiter=",")
        except ValueError:
            region = np.zeros(self.shape, dtype=np.uint8)
            likely_map = np.zeros(self.shape)

        # mask gen
        mask = np.zeros((320, 320, 3))
        for i in range(1, region.max() + 1):
            peak_i = i % 20
            mask[region == i, 0] = r[peak_i] * 255
            mask[region == i, 1] = g[peak_i] * 255
            mask[region == i, 2] = b[peak_i] * 255
        # cv2.imwrite("mask.png", mask)

        gbs = []
        # each propagate
        peaks = np.insert(peaks, 0, [0, 0], axis=0)
        with open(file_path, mode="w") as f:
            f.write("ID,x,y\n")
            for i in range(region.max() + 1):
                f.write(f"{i},{peaks[i, 0]},{peaks[i ,1]}\n")
                mask = np.zeros(self.shape)
                mask[region == i] = likely_map[region == i]
                result = self.back_model(img, mask.astype(np.float32)).copy()

                result = result.clip(0, 255)

                savemat(
                    str(self.save_path.joinpath("{:04d}.mat".format(i))),
                    {"image": result, "mask": mask},
                )
                cv2.imwrite(
                    str(self.save_path.joinpath("{:04d}.png".format(i))), result
                )
                gbs.append(result)

            gbs_coloring = self.coloring(gbs)

            # mask gen
            gbs_coloring = np.array(gbs_coloring)
            index = np.argmax(gbs, axis=0)
            masks = np.zeros((320, 320, 3))
            for x in range(1, index.max() + 1):
                # mask = np.zeros((320, 320, 3))
                # mask[index == x, :] = gbs_coloring[x][index == x, :]
                masks[index == x, :] = gbs_coloring[x][index == x, :]

            cv2.imwrite(str(self.save_path.parent.joinpath("instance_backprop.png")), masks)

            gbs = np.array(gbs)
            gbs = (gbs / gbs.max() * 255).astype(np.uint8)

            # for i, gb in enumerate(gbs):
            # cv2.imwrite(str(save_path.joinpath("{:04d}.tif".format(i))), gb)
            cv2.imwrite(
                str(self.save_path.parent.joinpath("backward.tif")), gbs.max(axis=0)
            )

    def main(self):
        for img_i, path in enumerate(self.input_path):
            self.output_path_each = self.output_path.joinpath("{:05d}".format(img_i))
            self.output_path_each.mkdir(parents=True, exist_ok=True)

            img = np.array(Image.open(path))
            cv2.imwrite(str(self.output_path_each.joinpath("original.tif")), img)
            self.shape = img.shape
            if self.norm:
                img = (img.astype(np.float32) / 255).reshape(
                    (1, 1, img.shape[0], img.shape[1])
                )
            else:
                img = (img.astype(np.float32) / (255 / 2) - 1).reshape(
                    (1, 1, img.shape[0], img.shape[1])
                )
            img = torch.from_numpy(img)

            pre_img = self.unet_pred(
                img, self.output_path_each.joinpath("detection.tif")
            )

            img.requires_grad = True
            self.calculate(img, pre_img)
