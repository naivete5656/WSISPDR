import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class GuidedBackpropReLU(Function):
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(
            torch.zeros(input.size()).type_as(input), input, positive_mask
        )
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        imgs = []
        #grads = grad_output.detach().cpu().data.numpy()
        #if grads.max() != 0:
        #    for img in grads[0]:
        #        img = img/img.max()
        #        imgs.append(img)
        #    imgs = np.array(imgs)
        #    np.save(f'../back/{datetime.now().microsecond}.npy', imgs)

        input, output = self.saved_tensors

        positive_mask_1 = (input > 0).type_as(grad_output)

        #
        # import pdb
        #
        # pdb.set_trace()
        #grads = positive_mask_1.detach().cpu().data.numpy()
        #imgs = []
        #if grads.max() != 0:
        #    for img in grads[0]:
        #        img = img/img.max()
        ##        imgs.append(img)
        #    imgs = np.array(imgs)
        #    np.save(f'../back/relu-{datetime.now().microsecond}.npy', imgs)

        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            torch.addcmul(
                torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1
            ),
            positive_mask_2,
        )

        return grad_input


class GuidedBackpropReLU2(Function):
    @staticmethod
    def forward(ctx, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(
            torch.zeros(input.size()).type_as(input), input, positive_mask
        )
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors

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


def guide_relu(self, input):
    output = GuidedBackpropReLU2.apply(input)
    return output


class GuidedBack(nn.Module):
    def forward(self, input):
        return GuidedBackpropReLU


class GuidedBackpropMaxPool(Function):
    def forward(self, input):
        output, indices = F.max_pool2d(input, 2, return_indices=True)
        self.save_for_backward(indices)
        return output

    def backward(self, grad_output):
        # import pdb
        #
        # pdb.set_trace()
        indices = self.saved_tensors[0]
        unpooled = F.max_unpool2d(grad_output, indices, (2, 2))
        return unpooled
