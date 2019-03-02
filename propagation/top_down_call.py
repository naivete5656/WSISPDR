from call_backprop import BackProp
from gradcam import TopDownAfterReLu, TopDownBackprop
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
            for i, prm in enumerate(prms):
                cv2.imwrite(
                    str(each_back.joinpath(f"{i:05d}.tif")), prm.astype(np.uint8)
                )
            prm = np.max(prms, axis=0)
            plt.imshow(prm), plt.show()
            cv2.imwrite(str(all_back.joinpath(f"{i:05d}.tif")), prm.astype(np.uint8))


class TopDown(BackProp):
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
                prm = prm / prm.max() * 10
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
            cv2.imwrite(str(all_back.joinpath(f"{img_i:05d}.tif")), prm.astype(np.uint8))
