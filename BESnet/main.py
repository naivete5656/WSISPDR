import os
from pathlib import Path
import torch
from datetime import datetime
os.chdir(Path.cwd().parent)
from networks import UNetMultiTask2
from BESnet import TrainNet


if __name__ == "__main__":
    if __name__ == "__main__":
        torch.cuda.set_device(0)
        mode = "single"
        plot_size = 12
        date = datetime.now().date()
        # train_path = Path("./images/train")
        train_path = Path("/home/kazuya/file_server2/images/for_besnet/sequ9")
        # val_path = Path("./images/val")
        save_weight_path = Path("/home/kazuya/file_server2/weights/multi_task/sequ9/best.pth")

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
            plot_size=plot_size
        )

        train.fit()
