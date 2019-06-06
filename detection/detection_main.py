import torch
from datetime import datetime
from pathlib import Path

if __name__ == "__main__":
    import os

    os.chdir(Path.cwd().parent)
from detection.detection_train import TrainNet
from networks import UNet, UnetMultiFixedWeight, DilatedUNet
import torch
from datetime import datetime
from pathlib import Path
from detection.detection_train import TrainNet
from networks import UNet, UnetMultiFixedWeight, DilatedUNet

torch.cuda.set_device(0)
mode = "single"
plot_size = 9
norm = True

for cross in range(1, 5):
    date = datetime.now().date()
    dataset = ['GBM', 'B23P17', 'elmer']
    dataset = dataset[1]
    dirs = sorted(
        Path("/home/kazuya/file_server2/miccai/images/{}_cut".format(dataset)).iterdir()
    )
    dirs.pop(cross)
    val_path = dirs.pop()
    train_path = dirs

    save_weight_path = Path(
        "/home/kazuya/file_server2/miccai/weights/{}/{}/best_{}.pth".format(
            dataset, cross, plot_size
        )
    )

    models = {"single": UNet, "multi": UnetMultiFixedWeight, "dilated": DilatedUNet}
    net = models[mode](n_channels=1, n_classes=1, sig=norm)

    # pre_trained_path = "./weights/MSELoss/best_{}.pth".format(plot_size)
    # if pre_trained_path is not None:
    #     net.load_state_dict(
    #         torch.load(pre_trained_path, map_location={"cuda:2": "cuda:1"})
    #     )

    net.cuda()

    train = TrainNet(
        net=net,
        mode=mode,
        epochs=500,
        batch_size=17,
        lr=1e-3,
        gpu=True,
        plot_size=plot_size,
        train_path=train_path,
        val_path=val_path,
        weight_path=save_weight_path,
        norm=norm,
    )

    train.main()
