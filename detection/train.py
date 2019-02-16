from tqdm import tqdm
from torch import optim
from eval import *
import os
import sys

path = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(path)

from networks import *
from utils import get_imgs_and_masks, batch, get_imgs_internal, get_imgs_multi
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


class TrainNet:
    def __init__(
            self,
            net,
            mode,
            epochs,
            batch_size,
            lr,
            gpu,
            plot_size,
            train_path,
            val_path,
            weight_path,
    ):
        self.net = net
        self.mode = mode
        self.ori_img_path = train_path / Path("ori")
        self.mask_path = train_path / Path("{}".format(plot_size))
        self.val_path = val_path / Path("ori")
        self.val_mask_path = val_path / Path("{}".format(plot_size))
        self.save_weight_path = weight_path
        self.save_weight_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            "Starting training:\nEpochs: {}\nBatch size: {} \nLearning rate: {}\ngpu:{}\n"
            "plot_size:{}\nmode:{}".format(epochs, batch_size, lr, gpu, plot_size, mode)
        )

        self.train = None
        self.val = None

        self.N_train = len(list(self.ori_img_path.glob("*.tif")))
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

    def show_graph(self):
        print("f-measure={}".format(max(self.evals)))
        x = list(range(len(self.losses)))
        plt.plot(x, self.losses)
        plt.plot(x, self.val_losses)
        plt.show()
        plt.plot(x, self.evals)
        plt.show()

    def validation(self, total_epoch):
        print("Epoch finished ! Loss: {}".format(self.epoch_loss / total_epoch))
        self.losses.append(self.epoch_loss / total_epoch)
        if 1:
            fmeasure, val_loss = eval_net(self.net, self.val, "single", gpu=self.gpu)
        print("f-measure: {}".format(fmeasure))
        print("val_loss: {}".format(val_loss))
        try:
            if max(self.evals) < fmeasure:
                torch.save(self.net.state_dict(), str(self.save_weight_path))
                self.bad = 0
            elif max(self.val_losses) > val_loss:
                pass
            else:
                self.bad += 1
        except ValueError:
            torch.save(self.net.state_dict(), str(self.save_weight_path))
        self.evals.append(fmeasure)
        self.val_losses.append(val_loss)

        print("bad = {}".format(self.bad))

    def load(self):
        self.net.train()
        # reset the generators
        self.train = get_imgs_and_masks(self.ori_img_path, self.mask_path)
        self.val = get_imgs_and_masks(self.val_path, self.val_mask_path)

    def main(self):
        for epoch in range(self.epochs):
            print("Starting epoch {}/{}.".format(epoch + 1, self.epochs))
            self.load()

            pbar = tqdm(total=self.N_train)
            for i, b in enumerate(batch(self.train, self.batch_size)):
                imgs = np.array([i[0] for i in b])
                true_masks = np.array([i[1] for i in b])

                imgs = torch.from_numpy(imgs)
                true_masks = torch.from_numpy(true_masks)

                if self.gpu:
                    imgs = imgs.cuda()
                    true_masks = true_masks.cuda()

                masks_pred = self.net(imgs)

                masks_probs_flat = masks_pred.view(-1)
                true_masks_flat = true_masks.view(-1)

                loss = self.criterion(masks_probs_flat, true_masks_flat)
                self.epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.update(self.batch_size)
            pbar.close()
            self.validation(i)
            if self.bad >= 50:
                print("stop running")
                break
        self.show_graph()


class TrainMulti(TrainNet):
    def __init__(
            self,
            net,
            epochs,
            batch_size,
            lr,
            gpu,
            plot_size,
            train_path,
            val_path,
            weight_path,
    ):
        super().__init__(
            net,
            mode,
            epochs,
            batch_size,
            lr,
            gpu,
            plot_size,
            train_path,
            val_path,
            weight_path,
        )
        self.ex1_path = train_path.joinpath("3")
        self.ex2_path = train_path.joinpath("6")
        self.ex3_path = train_path.joinpath("9")
        self.val_ex1_path = train_path.joinpath("3")
        self.val_ex2_path = train_path.joinpath("6")
        self.val_ex3_path = train_path.joinpath("9")
        self.net.ex1.load_state_dict(
            torch.load("./weight/MSELoss/best_3.pth", map_location={"cuda:0": "cpu"})
        )
        self.net.ex2.load_state_dict(
            torch.load("./weight/MSELoss/best_6.pth", map_location={"cuda:1": "cpu"})
        )
        self.net.ex3.load_state_dict(
            torch.load("./weight/MSELoss/best_9.pth", map_location={"cuda:2": "cpu"})
        )
        for param in self.net.ex1.parameters():
            param.requires_grad = False
        for param in self.net.ex2.parameters():
            param.requires_grad = False
        for param in self.net.ex3.parameters():
            param.requires_grad = False

    def load(self):
        self.net.train()

        # reset the generators
        self.train = get_imgs_multi(
            self.ori_img_path, self.ex1_path, self.ex2_path, self.ex3_path
        )
        self.val = get_imgs_multi(
            self.val_path, self.val_ex1_path, self.val_ex2_path, self.val_ex3_path
        )

    def main(self):
        for epoch in range(self.epochs):
            print("Starting epoch {}/{}.".format(epoch + 1, self.epochs))
            self.load()

            pbar = tqdm(total=self.N_train)
            for i, b in enumerate(batch(train, self.batch_size)):
                imgs = np.array([i[0] for i in b])
                ex1 = np.array([i[1] for i in b])
                ex2 = np.array([i[2] for i in b])
                ex3 = np.array([i[3] for i in b])

                imgs = torch.from_numpy(imgs)
                ex1 = torch.from_numpy(ex1)
                ex2 = torch.from_numpy(ex2)
                ex3 = torch.from_numpy(ex3)

                if self.gpu:
                    imgs = imgs.cuda()
                    ex1 = ex1.cuda()
                    ex2 = ex2.cuda()
                    ex3 = ex3.cuda()

                pre_ex1, pre_ex2, pre_ex3, res = net(imgs)

                loss1 = self.criterion(pre_ex1, ex1)
                loss2 = self.criterion(pre_ex2, ex2)
                loss3 = self.criterion(pre_ex3, ex3)
                loss4 = self.criterion(res, ex1)
                loss = loss1 + loss2 + loss3 + loss4
                self.epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.update(self.batch_size)
            pbar.close()
            self.validation(i)
            if self.bad >= 15:
                print("stop running")
                break
            print("bad = {}".format(self.bad))
        self.show_graph()


if __name__ == "__main__":
    torch.cuda.set_device(1)
    mode = "single"
    plot_size = 12
    date = datetime.now().date()
    train_path = [Path("../images/sequ_cut/sequ9"), Path("../images/sequ_cut/sequ17")]
    val_path = Path("../images/challenge_cut/val_center")
    save_weight_path = Path("../weight/{}/challenge/best_{}.pth".format(date, plot_size))

    models = {"single": UNet, "multi": UnetMultiFixedWeight}
    net = models[mode](n_channels=1, n_classes=1)
    net.cuda()

    train = TrainNet(
        net=net,
        mode=mode,
        epochs=500,
        batch_size=10,
        lr=1e-5,
        gpu=True,
        plot_size=plot_size,
        train_path=train_path,
        val_path=val_path,
        weight_path=save_weight_path,
    )

    train.main()
