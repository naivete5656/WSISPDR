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
                self.validation(i, epoch)
            else:
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


# if __name__ == "__main__":
#
#     torch.cuda.set_device(1)
#     # mode = "single"
#     mode = "single"
#     plot_size = 3
#     norm = True
#     date = datetime.now().date()
#     train_path = Path("./images/C2C12P7/sequ_cut/0318/sequ17")
#     val_path = Path("./images/C2C12P7/sequ_cut/0318/sequ18")
#     save_weight_path = Path("/home/kazuya/file_server2/weights/marge/best_{}.pth".format(plot_size))
#     save_path = Path("./detection/confirm")
#     pre_trained_path = './weights/MSELoss/best_12.pth'
#     # pre_trained_path = ''
#     models = {"single": MargeNet}
#     net = UNet(n_channels=1, n_classes=1, sig=norm)
#     if pre_trained_path is not None:
#         net.load_state_dict(torch.load(pre_trained_path, map_location={"cuda:3": "cuda:1"}))
#     net = MargeNet(n_channels=1, n_classes=1, sig=norm, net=net)
#
#     net.cuda()
#
#     train = TrainMarge(
#         net=net,
#         mode=mode,
#         epochs=500,
#         batch_size=5,
#         lr=1e-5,
#         gpu=True,
#         plot_size=plot_size,
#         train_path=train_path,
#         val_path=val_path,
#         weight_path=save_weight_path,
#     )
#
#     train.main()
