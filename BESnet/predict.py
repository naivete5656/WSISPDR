from pathlib import Path
import torch
import numpy as np
import os

os.chdir(Path.cwd().parent)
from networks import UNetMultiTask
from datetime import datetime
from PIL import Image
import cv2


class Predict:
    def __init__(self, net, gpu, weight_path, root_path, save_path, normvalue):
        self.peak_thresh = 100
        self.dist_peak = 2
        self.dist_threshold = 10

        self.net = net
        self.gpu = gpu

        self.weight_path = weight_path
        self.ori_path = root_path

        self.save_ori_path = save_path / Path("ori")
        self.save_pred_mask_path = save_path / Path("mask_pred")
        self.save_pred_boundary_path = save_path / Path("boundary_pred")

        self.save_ori_path.mkdir(parents=True, exist_ok=True)
        self.save_pred_mask_path.mkdir(parents=True, exist_ok=True)
        self.save_pred_boundary_path.mkdir(parents=True, exist_ok=True)

        self.normvalue = normvalue

    def main(self):
        net.load_state_dict(
            torch.load(self.weight_path, map_location={"cuda:3": "cuda:1"})
        )

        losses = 0
        # path def
        path_x = sorted(self.ori_path.glob("*.tif"))

        net.eval()
        for i, b in enumerate(path_x):
            ori = np.array(Image.open(b))
            img = (ori.astype(np.float32) / self.normvalue).reshape(
                (1, ori.shape[0], ori.shape[1])
            )
            with torch.no_grad():
                img = torch.from_numpy(img).unsqueeze(0)
                if self.gpu:
                    img = img.cuda()
                mask_pred, boundary_pred = net(img)
            pre_img = mask_pred.detach().cpu().numpy()[0, 0]
            pre_boundary = boundary_pred.detach().cpu().numpy()[0, 0]
            pre_img = (pre_img * 255).astype(np.uint8)
            pre_boundary = (pre_boundary * 255).astype(np.uint8)

            cv2.imwrite(str(self.save_pred_mask_path / Path("%05d.tif" % i)), pre_img)
            cv2.imwrite(
                str(self.save_pred_boundary_path / Path("%05d.tif" % i)), pre_boundary
            )
            cv2.imwrite(str(self.save_ori_path / Path("%05d.tif" % i)), ori)


if __name__ == "__main__":
    date = datetime.now().date()
    gpu = True
    plot_size = 6
    key = 1
    net = UNetMultiTask(n_channels=1, n_classes=1)
    net.cuda()

    weight_path = "./weights/2019-01-17/multi_task/best.pth"
    root_path = Path("./images/a/cut")
    # root_path = Path("./images/train/ori")
    save_path = Path("./outputs/{}/".format(date))

    pred = Predict(
        net=net,
        gpu=gpu,
        weight_path=weight_path,
        root_path=root_path,
        save_path=save_path,
        normvalue=255,
    )

    pred.main()
