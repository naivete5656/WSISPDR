import torch
from tqdm import tqdm

from utils import *
from datetime import datetime
from networks import UNet, MargeNet
from detection import TrainNet


class TrainMarge(TrainNet):
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
        save_path,
    ):
        super().__init__(
            net=net,
            mode=mode,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            gpu=gpu,
            plot_size=plot_size,
            train_path=train_path,
            weight_path=weight_path,
            norm=True,
            val_path=val_path,
            save_path=save_path
        )

    def main(self):
        for epoch in range(self.epochs):
            print("Starting epoch {}/{}.".format(epoch + 1, self.epochs))
            self.load()

            pbar = tqdm(total=self.N_train)
            for i, b in enumerate(batch(self.train, self.batch_size)):
                imgs = np.array([i[0] for i in b])
                imgs2 = np.array([i[0] for i in b])
                targets = np.array([i[1] for i in b])

                imgs = torch.from_numpy(imgs)
                imgs2 = torch.from_numpy(imgs2)
                targets = torch.from_numpy(targets)

                if self.gpu:
                    imgs = imgs.cuda()
                    imgs2 = imgs2.cuda()
                    targets = targets.cuda()

                pre_imgs = self.net(imgs, imgs2)

                loss = self.criterion(pre_imgs, targets)
                self.epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.update(self.batch_size)
            pbar.close()
            if self.val_path is not None:
                self.validation(i, epoch, mode='marge')
            else:
                torch.save(
                    self.net.state_dict(),
                    str(
                        self.save_weight_path.parent.joinpath(
                            "epoch_weight/{:05d}.pth".format(epoch)
                        )
                    ),
                )

            torch.save(
                self.net.state_dict(),
                str(
                    self.save_weight_path.parent.joinpath(
                        "epoch_weight/{:05d}.pth".format(epoch)
                    )
                ),
            )

            if self.bad >= 50:
                print("stop running")
                break
        self.show_graph()



