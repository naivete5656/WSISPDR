from .gen_guided_model import GuidedModel
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy.io import savemat


class BackProp(object):
    def __init__(self, input_path, output_path, net, gpu=True, sig=True):
        self.input_path = input_path
        self.output_path = output_path
        output_path.mkdir(parents=True, exist_ok=True)

        self.gpu = gpu
        # network load
        self.net = net
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
        r, g, b = np.loadtxt("./utils/color.csv", delimiter=",")
        gbs_coloring = []
        for peak_i, gb in enumerate(gbs):
            gb = gb / gb.max() * 255
            gb = gb.clip(0, 255).astype(np.uint8)
            result = np.ones((self.shape[0], self.shape[1], 3))
            result = gb[..., np.newaxis] * result
            peak_i = peak_i % 20
            result[..., 0][result[..., 0] != 0] = r[peak_i] * gb[gb != 0]
            result[..., 1][result[..., 1] != 0] = g[peak_i] * gb[gb != 0]
            result[..., 2][result[..., 2] != 0] = b[peak_i] * gb[gb != 0]
            gbs_coloring.append(result)
        return gbs_coloring


class GuideCall(BackProp):
    def __init__(self, img_path, output_path, weight_path, gpu=True):
        super().__init__(img_path, output_path, weight_path, gpu)

        self.back_model = GuidedModel(self.net)
        self.back_model.inference()
        self.shape = None
        self.output_path_each = None

    def main(self):
        for img_i, path in enumerate(self.input_path):
            self.output_path_each = self.output_path.joinpath("{:05d}".format(img_i))
            self.output_path_each.mkdir(parents=True, exist_ok=True)

            # load image
            img = cv2.imread(str(path), 0)[:512, :512]
            self.shape = img.shape
            cv2.imwrite(str(self.output_path_each.joinpath("original.tif")), img)

            img = (img.astype(np.float32) / 255).reshape(
                (1, 1, img.shape[0], img.shape[1])
            )
            img = torch.from_numpy(img)

            # throw unet
            if self.gpu:
                img = img.cuda()

            module = self.back_model
            prms = module(img, self.output_path_each)

            prms = np.array(prms)

            r, g, b = np.loadtxt("./utils/color.csv", delimiter=",")
            prms_coloring = []

            for peak_i, prm in enumerate(prms):
                prm = prm / prm.max() * 255
                prm = prm.clip(0, 255).astype(np.uint8)
                result = np.ones((self.shape[0], self.shape[1], 3))
                result = prm[..., np.newaxis] * result
                peak_i = peak_i % 20
                result[..., 0][result[..., 0] != 0] = r[peak_i] * prm[prm != 0]
                result[..., 1][result[..., 1] != 0] = g[peak_i] * prm[prm != 0]
                result[..., 2][result[..., 2] != 0] = b[peak_i] * prm[prm != 0]
                prms_coloring.append(result)

            prms_coloring = np.array(prms_coloring)

            savemat(
                str(self.output_path_each.joinpath("prms.mat")),
                {"prms": prms, "color": prms_coloring},
            )

            prms_coloring = np.max(prms_coloring, axis=0)

            prms_coloring = (
                prms_coloring.astype(np.float) / prms_coloring.max() * 255
            ).astype(np.uint8)

            # plt.imshow(prms_coloring), plt.show()
            cv2.imwrite(
                str(self.output_path_each.joinpath("instance.tif")),
                prms_coloring.astype(np.uint8),
            )
