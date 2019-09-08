from pathlib import Path
from datetime import datetime
from propagation import GuideCall
from pathlib import Path
import torch
from networks import UNet

datasets = {
    1: "GBM",
    2: "B23P17",
    3: "elmer",
    4: "challenge",
    5: "0318_9",
    6: "sequence_10",
}
torch.cuda.set_device(0)


if __name__ == "__main__":
    gpu = True
    norm = True
    plot_size = 9
    radius = 1
    date = datetime.now().date()
    key = 0
    cross = 0
    dataset = datasets[3]
    input_path = sorted(
        Path("/home/kazuya/main/WSISPDR/image/riken/B2_1/ori").glob("*.tif")
    )
    output_path = Path("/home/kazuya/main/WSISPDR/output/riken/B2/guided")
    weight_path = "/home/kazuya/main/WSISPDR/weights/riken/best.pth"

    net = UNet(n_channels=1, n_classes=1)

    net.load_state_dict(torch.load(weight_path, map_location={"cuda:2": "cuda:0"}))

    bp = GuideCall(input_path, output_path, net)
    bp.main()
