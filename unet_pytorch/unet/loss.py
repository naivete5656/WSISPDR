import torch.nn as nn
import torch


class SignMseLoss(nn.Module):
    def __init__(self, plus_weight=1.3, minus_weight=0.7):
        super(SignMseLoss, self).__init__()
        self.plus = plus_weight
        self.minus = -minus_weight

    def forward(self, input, target):
        loss_plus = self.plus * (torch.sign(input - target) + 1) * ((input - target) ** 2)
        loss_minus = self.minus * (torch.sign(input - target) - 1) * ((input - target) ** 2)
        loss_plus = loss_plus.sum() / input.data.nelement()
        loss_minus = loss_minus.sum() / input.data.nelement()
        loss = loss_plus + loss_minus
        return loss


class MseLoss(nn.Module):
    def __init__(self, plus_weight=0.7, minus_weight=0.3):
        super(MseLoss, self).__init__()
        self.plus = plus_weight
        self.minus = -minus_weight

    def forward(self, input, target):
        return ((input - target) ** 2).sum() / input.data.nelement()


if __name__ == '__main__':
    x = torch.Tensor([2, 1, 0])
    y = torch.Tensor([1, 1, 1])
    a = SignMseLoss()
    loss = a(x, y)
    print('a=')
    print(loss)
    b = nn.MSELoss()
    # loss_mse = b(x, y)
    # print('b=')
    # print(loss_mse)
    loss.backward()
    print(loss)
