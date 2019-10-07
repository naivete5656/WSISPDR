from pathlib import Path
from datetime import datetime
from propagation import GuideCall
from pathlib import Path
import torch
from networks import UNet
import argparse


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="data path")
    parser.add_argument(
        "-i",
        "--input_path",
        dest="input_path",
        help="dataset's path",
        default="./image/test",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        dest="output_path",
        help="output path",
        default="./output/guided",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weight_path",
        dest="weight_path",
        help="load weight path",
        default="./weights/best.pth",
    )
    parser.add_argument(
        "-g", "--gpu", dest="gpu", help="whether use CUDA", action="store_true"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    args.input_path = sorted(Path(args.input_path).joinpath("ori").glob("*.png"))
    args.output_path = Path(args.output_path)

    net = UNet(n_channels=1, n_classes=1)
    net.load_state_dict(torch.load(args.weight_path, map_location="cpu"))
    args.net = net

    bp = GuideCall(args)
    bp.main()
