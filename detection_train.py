from tqdm import tqdm
from torch import optim
import torch.utils.data
from detection import *
from utils import CellImageLoad, batch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2


class _TrainBase:
    def __init__(
        self,
        net,
        epochs,
        batch_size,
        lr,
        gpu,
        plot_size,
        train_path,
        weight_path,
        save_path=None,
        val_path=None,
    ):
        ori_paths = self.gather_path(train_path, "ori")
        gt_paths = self.gather_path(train_path, "gt")
        data_loader = CellImageLoad(ori_paths, gt_paths)
        self.train_dataset_loader = torch.utils.data.DataLoader(
            data_loader, batch_size=batch_size, shuffle=True, num_workers=0
        )
        self.number_of_traindata = data_loader.__len__()

        ori_paths = self.gather_path(val_path, "ori")
        gt_paths = self.gather_path(val_path, "gt")
        data_loader = CellImageLoad(ori_paths, gt_paths)
        self.val_loader = torch.utils.data.DataLoader(
            data_loader, batch_size=5, shuffle=False, num_workers=0
        )

        self.save_weight_path = weight_path
        self.save_weight_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_weight_path.parent.joinpath("epoch_weight").mkdir(
            parents=True, exist_ok=True
        )
        if save_path is not None:
            self.save_path = save_path
            self.save_path.mkdir(parents=True, exist_ok=True)
        print(
            "Starting training:\nEpochs: {}\nBatch size: {} \nLearning rate: {}\ngpu:{}\n"
            "plot_size:{}\n".format(epochs, batch_size, lr, gpu, plot_size)
        )

        self.net = net

        self.train = None
        self.val = None

        self.N_train = None
        self.optimizer = optim.Adam(net.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.gpu = gpu
        self.plot_size = plot_size
        self.criterion = nn.MSELoss()
        self.losses = []
        self.val_losses = []
        self.evals = []
        self.epoch_loss = 0
        self.bad = 0

    def gather_path(self, train_paths, mode):
        ori_paths = []
        for train_path in train_paths:
            ori_paths.extend(sorted(train_path.joinpath(mode).glob("*.tif")))
        return ori_paths

    def show_graph(self):
        x = list(range(len(self.losses)))
        plt.plot(x, self.losses)
        plt.plot(x, self.val_losses)
        plt.show()


class TrainNet(_TrainBase):
    def loss_calculate(self, masks_probs_flat, true_masks_flat):
        return self.criterion(masks_probs_flat, true_masks_flat)

    def main(self):
        for epoch in range(self.epochs):
            print("Starting epoch {}/{}.".format(epoch + 1, self.epochs))

            pbar = tqdm(total=self.number_of_traindata)
            for i, data in enumerate(self.train_dataset_loader):
                imgs = data["image"]
                true_masks = data["gt"]

                if self.gpu:
                    imgs = imgs.cuda()
                    true_masks = true_masks.cuda()

                masks_pred = self.net(imgs)

                masks_probs_flat = masks_pred.view(-1)
                true_masks_flat = true_masks.view(-1)

                loss = self.loss_calculate(masks_probs_flat, true_masks_flat)
                self.epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.update(self.batch_size)
            pbar.close()
            masks_pred = masks_pred.detach().cpu().numpy()
            cv2.imwrite("conf.tif", (masks_pred * 255).astype(np.uint8)[0, 0])
            self.validation(i, epoch)

            if self.bad >= 100:
                print("stop running")
                break
        self.show_graph()

    def validation(self, number_of_train_data, epoch):
        loss = self.epoch_loss / (number_of_train_data + 1)
        print("Epoch finished ! Loss: {}".format(loss))

        self.losses.append(loss)
        if epoch % 10 == 0:
            torch.save(
                self.net.state_dict(),
                str(
                    self.save_weight_path.parent.joinpath(
                        "epoch_weight/{:05d}.pth".format(epoch)
                    )
                ),
            )
        val_loss = eval_net(self.net, self.val_loader, gpu=self.gpu)
        if loss < 0.1:
            print("val_loss: {}".format(val_loss))
            try:
                if min(self.val_losses) > val_loss:
                    print("update best")
                    torch.save(self.net.state_dict(), str(self.save_weight_path))
                    self.bad = 0
                else:
                    self.bad += 1
                    print("bad ++")
            except ValueError:
                torch.save(self.net.state_dict(), str(self.save_weight_path))
            self.val_losses.append(val_loss)
        else:
            print("loss is too large. Continue train")
            self.val_losses.append(val_loss)
        print("bad = {}".format(self.bad))
        self.epoch_loss = 0


if __name__ == "__main__":
    plot_size = 6

    train_path = [Path("dir")]
    val_path = [Path("dir")]

    # save weight path
    save_weight_path = Path("./weights/{}/best.pth".format(dataset))

    # define model
    net = UNet(n_channels=1, n_classes=1)
    net.cuda()

    train = TrainNet(
        net=net,
        epochs=500,
        batch_size=16,
        lr=1e-3,
        gpu=True,
        plot_size=plot_size,
        train_path=train_path,
        val_path=val_path,
        weight_path=save_weight_path,
    )

    train.main()
