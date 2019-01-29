import torch
from torch.autograd import Function
import numpy as np
from skimage.draw import circle
from utils import *


class GuidedBackpropReLU(Function):
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(
            torch.zeros(input.size()).type_as(input), input, positive_mask
        )
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            torch.addcmul(
                torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1
            ),
            positive_mask_2,
        )

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model._modules.items():
            for idx1, module in module._modules.items():
                for idx2, module in module._modules.items():
                    for idx3, module in module._modules.items():
                        if module.__class__.__name__ == 'ReLU':
                            self.model._modules[idx]._modules[idx1]._modules[idx2]._modules[idx3] = GuidedBackpropReLU()
                        for idx4, module in module._modules.items():
                            if module.__class__.__name__ == 'ReLU':
                                self.model._modules[idx]._modules[idx1]._modules[idx2]._modules[idx3]._modules[
                                    idx4] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, x, y, radius, index=None):
        if input.grad is not None:
            input.grad.zero_()

        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        # make mask
        one_hot = np.zeros(output.shape[-2:], dtype=np.float32)
        rr, cc = circle(y, x, radius, one_hot.shape)
        one_hot[rr, cc] = 1

        one_hot = torch.from_numpy(one_hot)
        one_hot.requires_grad = True
        output = output.view(-1)
        one_hot = one_hot.view(-1)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        one_hot.backward(retain_graph=True)

        output = input.grad.detach().cpu().data.numpy()
        output = output[0, 0, :, :]

        return output
