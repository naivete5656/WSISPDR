from datetime import datetime
import torch
from pathlib import Path
from .detection import TrainNet
from .networks import UNet
from .propagation import GuideCall


if __name__ == "__main__":
    torch.cuda.set_device(1)

    date = datetime.now().date()
    gpu = True
    plot_size = 9
    key = 2

    weight_path = "./weights/best.pth"
    root_path = Path("./images/dataset/elmer_set/heavy1/ori")
    output_path = Path("output")

    net = UNet(n_channels=1, n_classes=1)
    net.cuda()
    net.load_state_dict(torch.load(weight_path, map_location={"cuda:3": "cuda:1"}))

    # image_path
    train_path = Path("./image/train")
    val_path = Path("./image/val")

    input_path = sorted(
        train_path[0].glob("*.tif")
    )

    # save weight path
    save_weight_path = Path("./weight/best.pth")

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

    net = UNet(n_channels=1, n_classes=1)

    net.load_state_dict(torch.load(weight_path, map_location={"cuda:2": "cuda:0"}))

    bp = GuideCall(input_path, output_path, net)
    bp.main()




