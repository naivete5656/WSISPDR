from types import MethodType
import torch.nn as nn
from .guided_parts import guide_relu
from utils import local_maxima, gaus_filter
from scipy.io import savemat
import torch
import numpy as np
import cv2


class GuidedModel(nn.Sequential):
    def __init__(self, *args, **kargs):
        super().__init__(*args)
        self.inferencing = False
        self.shape = None

    def _patch(self):
        for module in self.modules():
            if isinstance(module, nn.ReLU):
                module._original_forward = module.forward
                module.forward = MethodType(guide_relu, module)

    def _recover(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, "_original_forward"):
                module.forward = module._original_forward

    def forward(
        self,
        img,
        root_path,
        peak=None,
        class_threshold=0,
        peak_threshold=30,
        retrieval_cfg=None,
    ):
        assert img.dim() == 4, "PeakResponseMapping layer only supports batch mode."
        if self.inferencing:
            img.requires_grad_()

        # classification network forwarding
        class_response_maps = super().forward(img)
        # peak backpropagation
        # grad_output = mask

        pre_img = class_response_maps.detach().cpu().numpy()[0, 0]
        self.shape = pre_img.shape
        if peak is None:
            cv2.imwrite(
                str(root_path.joinpath("detection.tif")), (pre_img * 255).astype(np.uint8)
            )
        # peak
        peaks = local_maxima((pre_img * 255).astype(np.uint8), 50, 2).astype(np.int)

        gauses = []
        try:
            for peak in peaks:
                temp = np.zeros(self.shape)
                temp[peak[1], peak[0]] = 255
                gauses.append(gaus_filter(temp, 401, 12))
            region = np.argmax(gauses, axis=0) + 1
            likely_map = np.max(gauses, axis=0)
            region[likely_map < 0.01] = 0
            #
            # r, g, b = np.loadtxt("./utils/color.csv", delimiter=",")
        except ValueError:
            region = np.zeros(self.shape, dtype=np.uint8)
            likely_map = np.zeros(self.shape)

        gbs = []
        # each propagate
        peaks = np.insert(peaks, 0, [0, 0], axis=0)
        if peak is None:
            with open(root_path.joinpath("peaks.txt"), mode="w") as f:
                f.write("ID,x,y\n")
                for i in range(region.max() + 1):
                    if img.grad is not None:
                        img.grad.zero_()
                    # f.write(f"{i},{peaks[i, 0]},{peaks[i ,1]}\n")
                    f.write("{},{},{}\n".format(i, peaks[i, 0], peaks[i, 1]))
                    mask = np.zeros(self.shape, dtype=np.float32)
                    mask[region == i] = 1
                    mask = mask.reshape([1, 1, self.shape[0], self.shape[1]])
                    mask = torch.from_numpy(mask)
                    mask = mask.cuda()

                    class_response_maps.backward(mask, retain_graph=True)
                    result = img.grad.detach().sum(1).clone().clamp(min=0).cpu().numpy()

                    save_path = root_path.joinpath('each_peak')
                    save_path.mkdir(parents=True, exist_ok=True)
                    savemat(
                        str(save_path.joinpath("{:04d}.mat".format(i))),
                        {"image": result[0], "mask": mask},
                    )
                    cv2.imwrite(
                        str(save_path.joinpath("{:04d}.png".format(i))),
                        (result[0] * 255).astype(np.uint8),
                    )
                    gbs.append(result[0])

        else:
            for i in range(region.max() + 1):
                if img.grad is not None:
                    img.grad.zero_()
                mask = np.zeros(self.shape, dtype=np.float32)
                mask[region == i] = 1
                mask = mask.reshape([1, 1, self.shape[0], self.shape[1]])
                mask = torch.from_numpy(mask)
                mask = mask.cuda()

                class_response_maps.backward(mask, retain_graph=True)
                result = img.grad.detach().sum(1).clone().clamp(min=0).cpu().numpy()
                gbs.append(result[0])

        return gbs

    def train(self, mode=True):
        super().train(mode)
        if self.inferencing:
            self._recover()
            self.inferencing = False
        return self

    def inference(self):
        super().train(False)
        self._patch()
        self.inferencing = True
        return self



