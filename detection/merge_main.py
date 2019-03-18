import os
from pathlib import Path

import torch
from datetime import datetime

os.chdir(Path.cwd().parent)
from networks import MargeNet,UNet
from detection import TrainMarge


if __name__ == "__main__":

    torch.cuda.set_device(1)
    # mode = "single"
    mode = "single"
    plot_size = 3
    norm = True
    date = datetime.now().date()
    train_path = Path("./images/C2C12P7/sequ_cut/0318/sequ17")
    val_path = Path("./images/C2C12P7/sequ_cut/0318/sequ18")
    save_weight_path = Path("/home/kazuya/file_server2/weights/marge/best_{}.pth".format(plot_size))
    save_path = Path("./detection/confirm")
    pre_trained_path = '/home/kazuya/main/weakly_supervised_instance_segmentation/weights/MSELoss/best_12.pth'
    # pre_trained_path = ''
    models = {"single": MargeNet}
    net = UNet(n_channels=1, n_classes=1, sig=norm)
    if pre_trained_path is not None:
        net.load_state_dict(torch.load(pre_trained_path, map_location={"cuda:3": "cpu"}))
    net = MargeNet(n_channels=1, n_classes=1, sig=norm, net=net)

    # net.cuda()

    train = TrainMarge(
        net=net,
        mode=mode,
        epochs=500,
        batch_size=10,
        lr=1e-5,
        gpu=False,
        plot_size=plot_size,
        train_path=train_path,
        val_path=val_path,
        weight_path=save_weight_path,
        save_path=Path('./confirm')
    )

    train.main()
