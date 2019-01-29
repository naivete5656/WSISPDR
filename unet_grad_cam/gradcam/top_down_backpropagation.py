from types import MethodType

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.misc import imresize
from .pr_conv import pr_conv2d


import numpy as np
import cv2
from skimage.feature import peak_local_max


def local_maxima(img, threshold, dist):
    data = np.zeros((0, 2))
    x = peak_local_max(img, threshold_abs=threshold, min_distance=dist)
    peak_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for j in range(x.shape[0]):
        peak_img[x[j, 0], x[j, 1]] = 255
    labels, _, _, center = cv2.connectedComponentsWithStats(peak_img)
    for j in range(1, labels):
        data = np.append(data, [[center[j, 0], center[j, 1]]], axis=0)
    return data


class TopDownBackprop(nn.Sequential):
    def __init__(self, *args, **kargs):
        super(TopDownBackprop, self).__init__(*args)
        self.inferencing = False

    def _patch(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module._original_forward = module.forward
                module.forward = MethodType(pr_conv2d, module)

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

            plt.imshow(img[0]), plt.show()
            prms.append(input.grad.detach().sum(1).clone().clamp(min=0).cpu().numpy()[0])
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
