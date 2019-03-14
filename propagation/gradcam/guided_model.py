import torch
import torch.nn as nn
from .guided_parts import GuidedBackpropReLU, GuidedBackpropMaxPool
import numpy as np
from skimage.draw import circle


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

        one_hot.backward()

        output = input.grad.detach().cpu().data.numpy()
        output = output[0, 0, :, :]

        return output
