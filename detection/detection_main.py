from pathlib import Path

if __name__ == "__main__":
    import os

    os.chdir(Path.cwd().parent)
import torch
from datetime import datetime
from pathlib import Path
from detection.detection_train import TrainNet
from networks import UNet

torch.cuda.set_device(0)

for cross in range(4):
    plot_size = 9
    norm = True
    # dataset
    date = datetime.now().date()
    dataset = ['GBM', 'B23P17', 'elmer']
    dataset = dataset[2]
    dirs = sorted(
        Path("/home/kazuya/ssd/weakly_supervised_instance_segmentation/images/data/{}_cut".format(dataset)).iterdir()
    )
    dirs.pop(cross)

    # choose val data in 4 sequence
    val_path = dirs.pop()

    train_path = dirs

    # save weight path
    save_weight_path = Path(
        "/home/kazuya/file_server2/miccai/weights/{}/{}/best_{}.pth".format(
            dataset, cross, plot_size
        )
    )

    # define model
    net = UNet(n_channels=1, n_classes=1)
    net.cuda()

    train = TrainNet(
        net=net,
        epochs=500,
        batch_size=17,
        lr=1e-3,
        gpu=True,
        plot_size=plot_size,
        train_path=train_path,
        val_path=val_path,
        weight_path=save_weight_path,
    )

    train.main()
