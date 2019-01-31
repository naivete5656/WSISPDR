from pathlib import Path
from gradcam import *
import torch
from unet import UNet
from PIL import Image
import numpy as np
import cv2
from utils import plot_3d, local_maxima, gaus_filter
import matplotlib.pyplot as plt
from scipy.io import savemat


class _BackProp(object):
    def __init__(self, input_path, output_path, weight_path, gpu=True):
        self.input_path = input_path
        self.output_path = output_path
        output_path.mkdir(parents=True, exist_ok=True)

        self.gpu = gpu
        # network load
        self.net = UNet(n_channels=1, n_classes=1)
        self.net.load_state_dict(
            torch.load(weight_path, map_location={"cuda:3": "cuda:0"})
        )
        self.net.eval()
        if self.gpu:
            self.net.cuda()

        self.shape = None

    def unet_pred(self, img, save_path=None):
        # throw unet
        if self.gpu:
            img = img.cuda()
        pre_img = self.net(img)
        pre_img = pre_img.detach().cpu().numpy()[0, 0]
        if save_path is not None:
            cv2.imwrite(str(save_path), (pre_img * 255).astype(np.uint8))
        return pre_img

    def main(self):
        for img_i, path in enumerate(self.input_path):
            # self.temp_path = self.text_output.joinpath("{:05d}.txt".format(img_i))
            # with open(self.temp_path, mode="w") as f:
            #     f.write("ID,x,y\n")
            # load image
            img = np.array(Image.open(path))
            self.shape = img.shape
            img = (img.astype(np.float32) / 255).reshape(
                (1, 1, img.shape[0], img.shape[1])
            )
            img = torch.from_numpy(img)

            pre_img = self.unet_pred(
                img, self.likelymap_savepath.joinpath(f"{img_i:05d}.tif")
            )

            img.requires_grad = True
            gbs = self.calculate(img, pre_img)

            self.save_img(gbs, img_i)


class TopDown(_BackProp):
    def __init__(self, input_path, output_path, weight_path, gpu=True, radius=1):
        super().__init__(input_path, output_path, weight_path, gpu)
        self.back_model = TopDownAfterReLu(self.net)
        self.back_model.inference()
        print(self)

    def main(self):
        each_back = self.output_path.joinpath("each")
        each_back.mkdir(parents=True, exist_ok=True)
        all_back = self.output_path.joinpath("all")
        all_back.mkdir(parents=True, exist_ok=True)
        for img_i, path in enumerate(self.input_path):
            # load image
            img = np.array(Image.open(path))
            img = (img.astype(np.float32) / 255).reshape(
                (1, 1, img.shape[0], img.shape[1])
            )
            img = torch.from_numpy(img)

            # throw unet
            if self.gpu:
                img = img.cuda()
            module = self.back_model
            prms = module(img)
            prms = np.array(prms)
            prms = prms / prms.max() * 255
            r, g, b = np.loadtxt("./utils/color.csv", delimiter=",")
            prms_coloring = []

            for peak_i, prm in enumerate(prms):
                prm = prm/prm.max() * 10
                prm = prm.clip(0, 255).astype(np.uint8)
                result = np.ones((320, 320, 3))
                result = prm[..., np.newaxis] * result
                peak_i = peak_i % 20
                result[..., 0][result[..., 0] != 0] = r[peak_i] * prm[prm != 0]
                result[..., 1][result[..., 1] != 0] = g[peak_i] * prm[prm != 0]
                result[..., 2][result[..., 2] != 0] = b[peak_i] * prm[prm != 0]
                prms_coloring.append(result)

            prms_coloring = np.array(prms_coloring)
            index = np.argmax(prms, axis=0)
            mask = np.zeros((320, 320, 3))
            for x in range(1, index.max() + 1):
                mask[index == x, :] = prms_coloring[x][index == x, :]
            plt.imshow(mask), plt.show()
            prms_coloring = (
                    prms_coloring.astype(np.float) / prms_coloring.max() * 255
            ).astype(np.uint8)

            prms_coloring = np.max(prms_coloring, axis=0)

            prm = np.max(prms, axis=0)
            plt.imshow(prms_coloring), plt.show()
            cv2.imwrite(str(all_back.joinpath(f"{i:05d}.tif")), prm.astype(np.uint8))


class BackpropagationEachPeak(_BackProp):
    def __init__(self, input_path, output_path, weight_path, gpu=True, radius=1):
        super().__init__(input_path, output_path, weight_path, gpu)

        self.back_model = Test(self.net)
        self.per_1ch_path = output_path.parent / Path(f"r={radius}_1ch")
        self.per_1ch_path.mkdir(parents=True, exist_ok=True)
        self.likelymap_savepath = output_path.parent.joinpath("likelymap")
        self.likelymap_savepath.mkdir(parents=True, exist_ok=True)
        self.text_output = output_path.parent.joinpath("id")
        self.text_output.mkdir(parents=True, exist_ok=True)

    def save_img(self, gbs, i):
        # coloring
        try:
            r, g, b = np.loadtxt("./utils/color.csv", delimiter=",")
            gbs_coloring = []
            # normalize
            max_value = np.array(gbs).max()
            gbs = (gbs / max_value) * 1000
            gbs = gbs.clip(0, 255).astype(np.uint8)

            # colorling
            for peak_i, gb in enumerate(gbs):
                gb = gb * 255
                gb = gb.clip(0, 255).astype(np.uint8)
                result = np.ones((self.shape[0], self.shape[1], 3))
                result = gb[..., np.newaxis] * result
                peak_i = peak_i % 20
                result[..., 0][result[..., 0] != 0] = r[peak_i] * gb[gb != 0]
                result[..., 1][result[..., 1] != 0] = g[peak_i] * gb[gb != 0]
                result[..., 2][result[..., 2] != 0] = b[peak_i] * gb[gb != 0]
                # cv2.imwrite(
                #     str(self.per_1ch_path / Path("%04d-%04d.tif" % (i, peak_i))),
                #     (result / result.max() * 255).astype(np.uint8),
                # )
                # result = gb * 10
                gbs_coloring.append(result)

            gbs_coloring = (
                gbs_coloring.astype(np.float) / gbs_coloring.max() * 255
            ).astype(np.uint8)

            gbs_coloring = np.max(gbs_coloring, axis=2)
            cv2.imwrite(str(self.output_path / Path("%05d.tif" % i)), gbs_coloring)
        except ValueError:
            cv2.imwrite(
                str(self.output_path / Path("%05d.tif" % i)), np.zeros(self.shape)
            )

    def calculate(self, img, pre_img):
        # peak
        peaks = local_maxima((pre_img * 255).astype(np.uint8), 125, 2).astype(np.int)

        gbs = []
        target_index = None
        for i, peak in enumerate(peaks):
            with open(self.temp_path, mode="a") as f:
                f.write(str(i) + "," + str(peak[0]) + "," + str(peak[1]) + "\n")
            result = self.back_model(img, peak[0], peak[1]).copy()
            gbs.append(result)
        return gbs


class BackpropAll(_BackProp):
    def __init__(self, input_path, output_path, weight_path, gpu=True, radius=1):
        super().__init__(input_path, output_path, weight_path, gpu)

        self.back_model = Test2(self.net)
        self.likelymap_savepath = output_path.parent.joinpath("likelymap")
        self.likelymap_savepath.mkdir(parents=True, exist_ok=True)

    def calculate(self, img, pre_img):

        mask = np.zeros(pre_img.shape)
        mask[pre_img > 0.1] = 1

        mask_bkg = np.zeros(pre_img.shape)
        mask_bkg[pre_img < 0.1] = 1

        result_fore = self.back_model(img, mask.astype(np.float32)).copy()
        return result_fore

    def save_img(self, result, i):
        cv2.imwrite(
            str(self.output_path / Path("%05d.tif" % i)),
            (result / result.max() * 255).clip(0, 255).astype(np.uint8),
        )


class BackPropBackGround(_BackProp):
    def __init__(self, input_path, output_path, weight_path, gpu=True, radius=1):
        super().__init__(input_path, output_path, weight_path, gpu)
        self.output_path_each = None
        self.back_model = Test2(self.net)

    def calculate(self, img, pre_img):
        file_path = self.output_path_each.joinpath("peaks.txt")
        save_path = self.output_path_each.joinpath("each_peak")
        save_path.mkdir(parents=True, exist_ok=True)

        # peak
        peaks = local_maxima((pre_img * 255).astype(np.uint8), 125, 2).astype(np.int)
        gauses = []

        for peak in peaks:
            temp = np.zeros(self.shape)
            temp[peak[1], peak[0]] = 255
            gauses.append(gaus_filter(temp, 401, 12))
        region = np.argmax(gauses, axis=0) + 1
        likely_map = np.max(gauses, axis=0)
        region[likely_map < 0.05] = 0

        r, g, b = np.loadtxt("./utils/color.csv", delimiter=",")

        # mask gen
        mask = np.zeros((320, 320, 3))
        for i in range(1, region.max() + 1):
            mask[region == i, 0] = r[i] * 255
            mask[region == i, 1] = g[i] * 255
            mask[region == i, 2] = b[i] * 255
        cv2.imwrite('mask.png', mask)

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
                # savemat(
                #     str(save_path.joinpath("{:04d}.mat".format(i))),
                #     {"image": result, "mask": mask},
                # )
                gbs.append(result)

            # coloring
            gbs_coloring = []
            for peak_i, gb in enumerate(gbs):
                gb = gb * 255
                gb = gb.clip(0, 255).astype(np.uint8)
                result = np.ones((self.shape[0], self.shape[1], 3))
                result = gb[..., np.newaxis] * result
                peak_i = peak_i % 20
                result[..., 0][result[..., 0] != 0] = r[peak_i] * gb[gb != 0]
                result[..., 1][result[..., 1] != 0] = g[peak_i] * gb[gb != 0]
                result[..., 2][result[..., 2] != 0] = b[peak_i] * gb[gb != 0]
                gbs_coloring.append(result)

            # mask gen
            gbs_coloring = np.array(gbs_coloring)
            index = np.argmax(gbs, axis=0)
            mask = np.zeros((320, 320, 3))
            for x in range(1, index.max() + 1):
                mask[index == x, :] = gbs_coloring[x][index == x, :]
            cv2.imwrite('instance_backprop.png', mask)

            gbs = np.array(gbs)
            gbs = (gbs / gbs.max() * 255).astype(np.uint8)

            for i, gb in enumerate(gbs):
                cv2.imwrite(str(save_path.joinpath("{:04d}.tif".format(i))), gb)
            cv2.imwrite(
                str(save_path.parent.joinpath("backward.tif".format(i))),
                gbs.max(axis=0),
            )

    def main(self):
        for img_i, path in enumerate(self.input_path):
            self.output_path_each = self.output_path.joinpath("{:05d}".format(img_i))
            self.output_path_each.mkdir(parents=True, exist_ok=True)

            img = np.array(Image.open(path))
            self.shape = img.shape
            img = (img.astype(np.float32) / 255).reshape(
                (1, 1, img.shape[0], img.shape[1])
            )
            img = torch.from_numpy(img)

            pre_img = self.unet_pred(
                img, self.output_path_each.joinpath("detection.tif")
            )

            img.requires_grad = True
            gbs = self.calculate(img, pre_img)
