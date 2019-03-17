from .call_backprop import BackProp
from .gradcam import TopDownAfterReLu, TopDownBackprop, GuidedModel
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


class TopDownBefore(BackProp):
    def __init__(self, input_path, output_path, weight_path, gpu=True, radius=1):
        super().__init__(input_path, output_path, weight_path, gpu)
        self.back_model = TopDownBackprop(self.net)
        self.back_model.inference()
        self.shape = None

    def main(self):
        each_back = self.output_path.joinpath("each")
        each_back.mkdir(parents=True, exist_ok=True)
        all_back = self.output_path.joinpath("all")
        all_back.mkdir(parents=True, exist_ok=True)
        for img_i, path in enumerate(self.input_path):
            # load image
            img = np.array(Image.open(path))
            self.shape = img.shape
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
            for i, prm in enumerate(prms):
                cv2.imwrite(
                    str(each_back.joinpath("{:05d}.tif".format(i))),
                    prm.astype(np.uint8),
                )
            prm = np.max(prms, axis=0)
            cv2.imwrite(
                str(all_back.joinpath("{:05d}.tif".format(i))), prm.astype(np.uint8)
            )


class TopDown(BackProp):
    def __init__(self, input_path, output_path, weight_path, gpu=True, radius=1):
        super().__init__(input_path, output_path, weight_path, gpu)
        self.back_model = TopDownAfterReLu(self.net)
        self.back_model.inference()
        self.shape = None

    def main(self):
        each_back = self.output_path.joinpath("each")
        each_back.mkdir(parents=True, exist_ok=True)
        all_back = self.output_path.joinpath("all")
        all_back.mkdir(parents=True, exist_ok=True)
        for img_i, path in enumerate(self.input_path):
            # load image
            img = np.array(Image.open(path))
            self.shape = img.shape
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
                prm = prm / prm.max() * 10
                prm = prm.clip(0, 255).astype(np.uint8)
                result = np.ones((self.shape[0], self.shape[1], 3))
                result = prm[..., np.newaxis] * result
                peak_i = peak_i % 20
                result[..., 0][result[..., 0] != 0] = r[peak_i] * prm[prm != 0]
                result[..., 1][result[..., 1] != 0] = g[peak_i] * prm[prm != 0]
                result[..., 2][result[..., 2] != 0] = b[peak_i] * prm[prm != 0]
                prms_coloring.append(result)

            prms_coloring = np.array(prms_coloring)
            index = np.argmax(prms, axis=0)
            mask = np.zeros((self.shape[0], self.shape[1], 3))
            for x in range(1, index.max() + 1):
                mask[index == x, :] = prms_coloring[x][index == x, :]
            plt.imshow(mask), plt.show()
            prms_coloring = (
                prms_coloring.astype(np.float) / prms_coloring.max() * 255
            ).astype(np.uint8)

            prms_coloring = np.max(prms_coloring, axis=0)

            prm = np.max(prms, axis=0)
            plt.imshow(prms_coloring), plt.show()
            cv2.imwrite(
                str(all_back.joinpath("{:05d}.tif".format(img_i))), prm.astype(np.uint8)
            )


class GuideCall(BackProp):
    def __init__(self, img_path, output_path, weight_path, gpu=True, radius=1):
        super().__init__(img_path, output_path, weight_path, gpu)
        self.back_model = GuidedModel(self.net)
        self.back_model.inference()
        self.shape = None
        self.output_path_each = None

    def main(self):
        for img_i, path in enumerate(self.input_path):
            self.output_path_each = self.output_path.joinpath(f"{img_i:05d}")
            self.output_path_each.mkdir(parents=True, exist_ok=True)

            # load image
            img = np.array(Image.open(path))
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
            # index = np.argmax(prms, axis=0)
            # mask = np.zeros((self.shape[0], self.shape[1], 3))
            # for x in range(1, index.max() + 1):
            #     mask[index == x, :] = prms_coloring[x][index == x, :]

            prms_coloring = np.max(prms_coloring, axis=0)

            prms_coloring = (
                prms_coloring.astype(np.float) / prms_coloring.max() * 255
            ).astype(np.uint8)

            prm = np.max(prms, axis=0)
            prm = prm / prm.max() * 255
            cv2.imwrite(
                str(self.output_path_each.joinpath("instance.tif")),
                prms_coloring.astype(np.uint8),
            )
            cv2.imwrite(
                str(self.output_path_each.joinpath("{:05d}.tif".format(img_i))),
                prm.astype(np.uint8),
            )

class GuideOnly(GuideCall):
    def main(self):
        save_paht = self.output_path.joinpath("instance")
        save_paht.mkdir(parents=True, exist_ok=True)

        for img_i, path in enumerate(self.input_path):

            # load image
            img = np.array(Image.open(path))
            self.shape = img.shape
            # cv2.imwrite(str(self.output_path.joinpath(f"ori/{img_i:05d}.tif")), img)

            img = (img.astype(np.float32) / 255).reshape(
                (1, 1, img.shape[0], img.shape[1])
            )
            img = torch.from_numpy(img)

            # throw unet
            if self.gpu:
                img = img.cuda()

            module = self.back_model
            prms = module(img, self.output_path_each, peak=1)

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
            # index = np.argmax(prms, axis=0)
            # mask = np.zeros((self.shape[0], self.shape[1], 3))
            # for x in range(1, index.max() + 1):
            #     mask[index == x, :] = prms_coloring[x][index == x, :]

            prms_coloring = np.max(prms_coloring, axis=0)

            prms_coloring = (
                prms_coloring.astype(np.float) / prms_coloring.max() * 255
            ).astype(np.uint8)

            prm = np.max(prms, axis=0)
            prm = prm / prm.max() * 255

            cv2.imwrite(
                str(save_paht.joinpath(f"{img_i:05d}.tif")),
                prms_coloring.astype(np.uint8),
            )
