import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from utils import gaus_filter
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
        help="txt path",
        default="./sample_cell_position.txt",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        dest="output_path",
        help="output path",
        default="./image/gt",
        type=str,
    )
    parser.add_argument(
        "-w", "--width", dest="width", help="image width", default=512, type=int
    )
    parser.add_argument(
        "-he", "--height", dest="height", help="height", default=512, type=int
    )
    parser.add_argument(
        "-g",
        "--gaussian_variance",
        dest="g_size",
        help="gaussian variance",
        default=12,
        type=int,
    )

    args = parser.parse_args()
    return args


def like_map_gen(args):

    args.output_path.mkdir(parents=True, exist_ok=True)
    # load txt file
    cell_positions = np.loadtxt(args.input_path, delimiter=",", skiprows=1)
    black = np.zeros((args.height, args.width))

    # 1013 - number of frame
    for i in range(0, int(cell_positions[:, 0].max())):
        # likelihood map of one input
        result = black.copy()
        cells = cell_positions[cell_positions[:, 0] == i]
        for _, x, y in cells:
            img_t = black.copy()  # likelihood map of one cell
            img_t[int(y)][int(x)] = 255  # plot a white dot
            img_t = gaus_filter(img_t, 301, args.g_size)
            result = np.maximum(result, img_t)  # compare result with gaussian_img
        #  normalization
        result = 255 * result / result.max()
        result = result.astype("uint8")
        cv2.imwrite(str(args.output_path / Path("%05d.tif" % i)), result)
        print(i + 1)
    print("finish")


if __name__ == "__main__":
    args = parse_args()
    args.output_path = Path(args.output_path)
    like_map_gen(args)
