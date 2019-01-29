import torch
from torch.autograd import Function
import torch.nn as nn
import numpy as np
from skimage.draw import circle
from utils import *
import torch.nn.functional as F


class GuidedBackpropReLU(Function):
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(
            torch.zeros(input.size()).type_as(input), input, positive_mask
        )
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        # import pdb
        #
        # pdb.set_trace()

        input, output = self.saved_tensors

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


class GuidedBackpropReLUModel(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.net = self.change(self.net)

    def change(self, module):
        if hasattr(module, "_modules") and len(module._modules) > 0:
            for key, item in module._modules.items():
                module._modules[key] = self.change(item)
            return module
        else:
            if isinstance(module, nn.ReLU):
                new = GuidedBackpropReLU()
            else:
                new = module
            return new

    def __call__(self, input, x, y, radius, index=None):
        if input.grad is not None:
            input.grad.zero_()

        if self.cuda:
            output = self.net.forward(input.cuda())
        else:
            output = self.net.forward(input)

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


class _BackpropReluCore(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        modules = list(self.net.modules())
        for module in modules:
            for i, child in enumerate(module.children()):
                if isinstance(child, nn.ReLU):
                    module._modules[str(i)] = GuidedBackpropReLU()

        self.shape = None

    def __call__(self, *args):
        input = args[0]

        self.shape = input.shape[-2:]

        if input.grad is not None:
            input.grad.zero_()

        if self.cuda:
            output = self.net.forward(input.cuda())
        else:
            output = self.net.forward(input)

        one_hot = self.make_mask(args[1:])
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

    def make_mask(self, args):
        pass


class Test(_BackpropReluCore):
    def make_mask(self, args):
        """
        :param args:(y,x,radius)
        :return: mask
        """
        # make mask
        one_hot = np.zeros(self.shape, dtype=np.float32)
        rr, cc = circle(args[1], args[0], 1, one_hot.shape)
        one_hot[rr, cc] = 1
        return one_hot


class Test2(_BackpropReluCore):
    def make_mask(self, args):
        # make mask1
        return args[0]


class Test3(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        modules = list(self.net.modules())
        for module in modules:
            for i, child in enumerate(module.children()):
                if isinstance(child, nn.ReLU):
                    module._modules[str(i)] = GuidedBackpropReLU()
                elif isinstance(child, nn.MaxPool2d):
                    module._modules[str(i)] = GuidedBackpropMaxPool()

    def make_mask(self, args):
        # make mask1
        return args[0]

    def __call__(self, *args):
        input = args[0]

        self.shape = input.shape[-2:]

        if input.grad is not None:
            input.grad.zero_()

        if self.cuda:
            output = self.net.forward(input.cuda())
        else:
            output = self.net.forward(input)

        one_hot = self.make_mask(args[1:])
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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(x)
        return x


if __name__ == "__main__":
    import os

    os.chdir("../")
    print(os.getcwd())

    a = torch.rand(1, 1, 64, 64)
    a.requires_grad = True
    pool = Net()
    output = pool(a)
    # maxpool = GuidedBackpropMaxPool()
    # maxpool.apply(a)
    print(output)
