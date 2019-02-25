from .train import _TrainBase
import torch
from tqdm import tqdm
from utils import *


class TrainMulti(_TrainBase):
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
            True,
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
            for i, b in enumerate(batch(self.train, self.batch_size)):
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

                pre_ex1, pre_ex2, pre_ex3, res = self.net(imgs)

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