from types import MethodType
import torch.nn as nn
import torch
from .pr_conv import pr_conv2d
from .guided_parts import guide_relu
from utils import local_maxima, gaus_filter

import numpy as np
import cv2
from skimage.feature import peak_local_max


class GuidedModel(nn.Sequential):
    def __init__(self, *args, **kargs):
        super().__init__(*args)
        self.inferencing = False

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
        self, input, peak=None, class_threshold=0, peak_threshold=30, retrieval_cfg=None
    ):
        assert input.dim() == 4, "PeakResponseMapping layer only supports batch mode."
        if self.inferencing:
            input.requires_grad_()

        # classification network forwarding
        class_response_maps = super().forward(input)
        # peak backpropagation
        # grad_output = mask
        grad_output = class_response_maps.new_empty(class_response_maps.size())

        pre_img = class_response_maps.detach().cpu().numpy()[0, 0]
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
            region[pre_img < 0] = 0

            r, g, b = np.loadtxt("./utils/color.csv", delimiter=",")
        except ValueError:
            region = np.zeros(self.shape, dtype=np.uint8)
            likely_map = np.zeros(self.shape)

        gbs = []
        # each propagate
        peaks = np.insert(peaks, 0, [0, 0], axis=0)
        with open(file_path, mode="w") as f:
            f.write("ID,x,y\n")
            for i in range(region.max() + 1):
                # f.write(f"{i},{peaks[i, 0]},{peaks[i ,1]}\n")
                f.write("{},{},{}\n".format(i, peaks[i, 0], peaks[i, 1]))
                mask = np.zeros(self.shape)
                mask[region == i] = likely_map[region == i]
                result = self.back_model(img, mask.astype(np.float32))

                result = result.clip(0, 255)

                savemat(
                    str(self.save_path.joinpath("{:04d}.mat".format(i))),
                    {"image": result, "mask": mask},
                )
                cv2.imwrite(
                    str(self.save_path.joinpath("{:04d}.png".format(i))), result
                )
                gbs.append(result)
        prms = []
        for peak in peaks:
            if input.grad is not None:
                input.grad.zero_()
            grad_output.zero_()
            # grad_output[0, 0, peak[1], peak[0]] = 1
            grad_output[0, 0, peak[1] - 5 : peak[1] + 5, peak[0] - 5 : peak[0] + 5] = 1
            class_response_maps.backward(grad_output, retain_graph=True)
            img = input.grad.detach().sum(1).clone().clamp(min=0).cpu().numpy()
            import matplotlib.pyplot as plt

            prms.append(
                input.grad.detach().sum(1).clone().clamp(min=0).cpu().numpy()[0]
            )
        return prms

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