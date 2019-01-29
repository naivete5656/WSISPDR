from pathlib import Path
import torch
import numpy as np
from networks import UNetMultiTask
from datetime import datetime
from PIL import Image
from utils import local_maxim, optimum, gt_mat_gen, remove_outside_plot, show_res
import cv2


class Predict:
    def __init__(self, net, gpu, weight_path, root_path, save_path, plot_size):
        self.peak_thresh = 100
        self.dist_peak = 2
        self.dist_threshold = 10

        self.net = net
        self.gpu = gpu

        self.weight_path = weight_path
        self.ori_path = root_path
        # self.gt_path = root_path / Path("{}".format(plot_size))

        self.save_ori_path = save_path / Path("ori")
        # self.save_gt_path = save_path / Path("gt")
        self.save_pred_mask_path = save_path / Path("mask_pred")
        self.save_pred_boundary_path = save_path / Path("boundary_pred")
        # self.save_error_path = save_path / Path("error")
        # self.save_txt_path = save_path / Path("f-measure.txt")

        self.save_ori_path.mkdir(parents=True, exist_ok=True)
        # self.save_gt_path.mkdir(parents=True, exist_ok=True)
        self.save_pred_mask_path.mkdir(parents=True, exist_ok=True)
        self.save_pred_boundary_path.mkdir(parents=True, exist_ok=True)
        # self.save_error_path.mkdir(parents=True, exist_ok=True)

    def main(self):
        net.load_state_dict(
            torch.load(self.weight_path, map_location={"cuda:3": "cuda:1"})
        )

        losses = 0
        # path def
        path_x = sorted(self.ori_path.glob("*.tif"))
        # path_y = sorted(self.gt_path.glob("*.tif"))

        # z = zip(path_x, path_y)
        """Evaluation without the densecrf with the dice coefficient"""
        net.eval()
        for i, b in enumerate(path_x):
            ori = np.array(Image.open(b))
            img = (ori.astype(np.float32) / 4096).reshape(
                (1, ori.shape[0], ori.shape[1])
            )
            # gt_img = np.array(Image.open(b[1]))
            with torch.no_grad():
                img = torch.from_numpy(img).unsqueeze(0)
                if self.gpu:
                    img = img.cuda()
                mask_pred, boundary_pred = net(img)
            pre_img = mask_pred.detach().cpu().numpy()[0, 0]
            pre_boundary = boundary_pred.detach().cpu().numpy()[0, 0]
            pre_img = (pre_img * 255).astype(np.uint8)
            pre_boundary = (pre_boundary * 255).astype(np.uint8)

            # gt = gt_mat_gen((gt_img).astype(np.uint8))
            # res = local_maxim(pre_img, self.peak_thresh, self.dist_peak)
            # associate_id = optimum(gt, res, self.dist_threshold)

            # gt_final, no_detected_id = remove_outside_plot(
            #     gt, associate_id, 0, pre_img.shape
            # )
            # res_final, overdetection_id = remove_outside_plot(
            #     res, associate_id, 1, pre_img.shape
            # )

            cv2.imwrite(str(self.save_pred_mask_path / Path("%05d.tif" % i)), pre_img)
            cv2.imwrite(str(self.save_pred_boundary_path / Path("%05d.tif" % i)), pre_boundary)
            cv2.imwrite(str(self.save_ori_path / Path("%05d.tif" % i)), ori)


if __name__ == "__main__":
    date = datetime.now().date()
    gpu = True
    plot_size = 6
    key = 1
    net = UNetMultiTask(n_channels=1, n_classes=1)
    net.cuda()

    weight_path = "./weights/2019-01-17/multi_task/best.pth"
    root_path = Path("./images/test")
    # root_path = Path("./images/train/ori")
    save_path = Path("./outputs/{}/test".format(date))

    pred = Predict(
        net=net,
        gpu=gpu,
        weight_path=weight_path,
        root_path=root_path,
        save_path=save_path,
        plot_size=plot_size,
    )

    pred.main()
