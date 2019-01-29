import torch
import torch.nn.functional as F
import torch
import torch.nn as nn

from torch.autograd import Function, Variable

# class InstanceDiceCoff(Function):


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def __init__(self):
        super().__init__()
        self.inter = None
        self.union = None

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = (
                grad_output
                * 2
                * (target * self.union - self.inter)
                / (self.union * self.union)
            )
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def eval_net(net, dataset, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    criterion = nn.BCELoss
    net.eval()
    tot = 0
    losses = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]
        true_boundary = b[2]

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        true_boundary = torch.from_numpy(true_boundary).unsqueeze(0)

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()
            true_boundary = true_boundary.cuda()

        mask_pred, boundary_pred = net(img)[0]

        loss1 = criterion(mask_pred, true_mask)
        loss2 = criterion(boundary_pred, true_boundary)
        loss = loss1 + loss2
        losses += loss.item()

        mask_pred = (mask_pred > 0.5).float()

        tot += dice_coeff(mask_pred, true_mask).item()
    return tot / i, losses / i


if __name__ == "__main__":
    print(1)
