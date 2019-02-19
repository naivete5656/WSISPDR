import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import *


def eval_net(
    net,
    dataset,
    mode,
    save_path=None,
    gpu=True,
    debug=False,
    dist_peak=2,
    peak_thresh=100,
    dist_threshold=10,
    only_loss=False,
):
    criterion = nn.MSELoss()
    net.eval()
    tps = 0
    fns = 0
    fps = 0
    losses = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]
        with torch.no_grad():
            img = torch.from_numpy(img).unsqueeze(0)
            gt = torch.from_numpy(true_mask).unsqueeze(0)
            if gpu:
                img = img.cuda()
                gt = gt.cuda()

            if mode == "single":
                mask_pred = net(img)
            else:
                _, _, _, mask_pred = net(img)
        pre_img = mask_pred.detach().cpu().numpy()[0, 0]
        pre_img = ((pre_img + 1) * (255 / 2)).astype(np.uint8)
        # if i == 1500:
        #     plt.subplot(131)
        #     plt.imshow(pre_img)
        #     plt.subplot(132)
        #     plt.imshow(b[0][0])
        #     plt.subplot(133)
        #     plt.imshow(b[1][0])
        #     plt.show()
        loss = criterion(mask_pred, gt)
        losses += loss.item()
        if not only_loss:

            gt = target_peaks_gen((true_mask[0] * 255).astype(np.uint8))
            res = local_maxima(pre_img, peak_thresh, dist_peak)
            associate_id = optimum(gt, res, dist_threshold)

            gt_final, no_detected_id = remove_outside_plot(
                gt, associate_id, 0, pre_img.shape
            )
            res_final, overdetection_id = remove_outside_plot(
                res, associate_id, 1, pre_img.shape
            )
            if debug:
                show_res(
                    b[0][0],
                    gt,
                    res,
                    no_detected_id,
                    overdetection_id,
                    path=str(save_path / Path("%05d.tif" % i)),
                )

            tp = associate_id.shape[0]
            fn = gt_final.shape[0] - associate_id.shape[0]
            fp = res_final.shape[0] - associate_id.shape[0]
            tps += tp
            fns += fn
            fps += fp
            if tps == 0:
                f_measure = 0
            else:
                recall = tps / (tps + fns)
                precision = tps / (tps + fps)
                f_measure = (2 * recall * precision) / (recall + precision)
    if only_loss:
        return losses / i
    else:
        return f_measure, (losses / i)
