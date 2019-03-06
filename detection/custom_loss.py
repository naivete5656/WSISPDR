import torch.nn as nn
import torch


class SignMseLoss(nn.Module):
    def __init__(self, weight_rate=1):
        super().__init__()
        self.weight = weight_rate

    def forward(self, input, target):
        target[target < 0.5] = 0
        input[target < 0.5] = 0
        return ((input - target) ** 2).sum() / input.data.nelement()


class MseLoss(nn.Module):
    def __init__(self, plus_weight=0.7, minus_weight=0.3):
        super(MseLoss, self).__init__()
        self.plus = plus_weight
        self.minus = -minus_weight

    def forward(self, input, target):
        return ((input - target) ** 2).sum() / input.data.nelement()


if __name__ == '__main__':
    x = torch.Tensor([1, 0.8, 0.1])
    y = torch.Tensor([1, 1, 0])
    a = SignMseLoss()
    loss = a(x, y)
    print('a=')
    print(loss)
    b = MseLoss()

    x = torch.Tensor([1, 0.8, 0.1])
    y = torch.Tensor([1, 1, 0])
    loss_mse = b(x, y)
    # print('b=')
    # print()
    # print(loss_mse)
    # loss.backward()
    print(loss_mse)
