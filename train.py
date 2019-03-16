import torch.nn as nn
from torch import optim
from datetime import datetime
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import *
from networks import UNetMultiTask2, BoundaryEnhancedCrossEntropyLoss
from eval import eval_net
from pathlib import Path


class TrainNet:
    def __init__(
        self, net, epochs, batch_size, lr, gpu, train_path, weight_path, val_path=None
    ):
        self.net = net

        self.ori_img_path = train_path / Path("ori")
        self.mask_path = train_path / Path("segmentation")
        self.boundary_path = train_path / Path("boundary")

        # self.val_path = val_path / Path("ori")
        # self.val_mask_path = val_path / Path("semantic")
        # self.val_boundary_path = val_path / Path("boundary")

        self.save_weight_path = weight_path
        self.save_weight_path.parent.mkdir(parents=True, exist_ok=True)

        self.N_train = len(list(self.ori_img_path.glob("*.tif")))
        self.optimizer = optim.Adam(net.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.gpu = gpu
        self.plot_size = plot_size
        self.criterion = nn.BCELoss()
        self.criterion2 = BoundaryEnhancedCrossEntropyLoss()
        self.val_losses = []
        self.evals = []
        print(
            "Starting training:\nEpochs: {}\nBatch size: {} \nLearning rate: {}\ngpu:{}\n".format(
                epochs, batch_size, lr, gpu
            )
        )

    def fit(self):
        losses = []

        bad = 0
        for epoch in range(self.epochs):
            print("Starting epoch {}/{}.".format(epoch + 1, self.epochs))
            self.net.train()

            # reset the generators
            train = get_imgs_and_masks_boundaries(
                self.ori_img_path, self.mask_path, self.boundary_path
            )
            # val = get_imgs_and_masks_boundaries(self.val_path, self.val_mask_path, self.val_boundary_path)

            epoch_loss = 0
            epoch_loss1 = 0
            epoch_loss2 = 0

            pbar = tqdm(total=self.N_train)
            for i, b in enumerate(batch(train, self.batch_size)):
                imgs = np.array([i[0] for i in b])
                true_masks = np.array([i[1] for i in b])
                true_boundaries = np.array([i[2] for i in b])

                imgs = torch.from_numpy(imgs)
                true_masks = torch.from_numpy(true_masks)
                true_boundaries = torch.from_numpy(true_boundaries)

                if self.gpu:
                    imgs = imgs.cuda()
                    true_masks = true_masks.cuda()
                    true_boundaries = true_boundaries.cuda()

                masks_pred, boundary_pred = self.net(imgs)

                masks_probs_flat = masks_pred.view(-1)
                true_masks_flat = true_masks.view(-1)
                loss1 = self.criterion(masks_probs_flat, true_masks_flat)

                boundary_probs_flat = boundary_pred.view(-1)
                true_boundary_flat = true_boundaries.view(-1)
                loss2 = self.criterion2(masks_probs_flat, boundary_probs_flat, true_boundary_flat)

                loss = loss1 + loss2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss1 += loss1.item()
                epoch_loss2 += loss2.item()
                epoch_loss += loss.item()

                pbar.update(self.batch_size)
            pbar.close()
            print("Epoch finished ! Loss: {}".format(epoch_loss / i))
            losses.append(epoch_loss / i)

            # bad = self.eval_append(val, bad)

            if bad >= 50:
                print("stop running")
                break
            print("bad = {}".format(bad))

            torch.save(self.net.state_dict(), str(self.save_weight_path))
        print("f_measure={}".format(max(self.evals)))

        x = list(range(len(losses)))
        plt.plot(x, losses)
        plt.plot(x, self.val_losses)
        plt.show()
        plt.plot(x, self.evals)
        plt.show()

    def eval_append(self, val, bad):
        if 1:
            val_dice, val_loss = eval_net(self.net, val, gpu=self.gpu)
        print("val_dice: {}".format(val_dice))
        print("val_loss: {}".format(val_loss))

        try:
            if max(self.evals) < val_dice:
                torch.save(self.net.state_dict(), str(self.save_weight_path))
                bad = 0
            elif min(self.val_losses) > val_loss:
                pass
            else:
                bad += 1
        except ValueError:
            torch.save(self.net.state_dict(), str(self.save_weight_path))

        self.evals.append(val_dice)
        self.val_losses.append(val_loss)
        return bad


if __name__ == "__main__":
    torch.cuda.set_device(1)
    mode = "single"
    plot_size = 12
    date = datetime.now().date()
    # train_path = Path("./images/train")
    train_path = Path("./images/train")
    # val_path = Path("./images/val")
    save_weight_path = Path("./weights/{}/multi_task/best.pth".format(date))

    network = UNetMultiTask2(n_channels=1, n_classes=1)
    network.cuda()

    train = TrainNet(
        net=network,
        epochs=500,
        batch_size=10,
        lr=1e-5,
        gpu=True,
        train_path=train_path,
        # val_path=val_path,
        weight_path=save_weight_path,
    )

    train.fit()
