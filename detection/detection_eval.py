import torch.nn as nn
import torch
import cv2
import numpy as np


def eval_net(
    net, dataset, gpu=True, vis=None, vis_im=None, vis_gt=None, loss=nn.MSELoss()
):
    criterion = loss
    net.eval()
    losses = 0
    torch.cuda.empty_cache()
    for iteration, data in enumerate(dataset):
        img = data["image"]
        target = data["gt"]
        if gpu:
            img = img.cuda()
            target = target.cuda()

        pred_img = net(img)

        loss = criterion(pred_img, target)
        losses += loss.data
    pred_img = pred_img.detach().cpu().numpy()
    cv2.imwrite("conf_eval.tif", (pred_img * 255).astype(np.uint8)[0, 0])
    return losses / iteration
