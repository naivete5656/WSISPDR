from datetime import datetime
import torch
from pathlib import Path
from detection import TrainNet
from networks import UNet
from propagation import GuideCall


if __name__ == "__main__":
    torch.cuda.set_device(1)

    date = datetime.now().date()
    gpu = True
    plot_size = 9
    key = 2

    weight_path = "./weight/best.pth"
    # image_path
    train_path = Path("./images/train")
    train_path = Path("/home/kazuya/main/weakly_supervised_instance_segmentation/images/sequ9")
    val_path = Path("./images/val")
    guided_input_path = sorted(
        train_path.joinpath("ori").glob("*.tif")
    )

    guided_input_path = guided_input_path[20000:]

    # guided output
    output_path = Path("output")

    # define model
    net = UNet(n_channels=1, n_classes=1)
    net.cuda()

    net.load_state_dict(torch.load(weight_path, map_location={"cuda:2": "cuda:0"}))

    bp = GuideCall(guided_input_path, output_path, net)
    bp.main()




