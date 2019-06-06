from scipy.io import loadmat
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


def coloring(gbs,save_path):
    # coloring
    r, g, b = np.loadtxt("../utils/color.csv", delimiter=",")
    gbs_coloring = []
    for peak_in, gb in enumerate(gbs):
        gb = gb / gb.max() * 13000
        gb = gb.clip(0, 255).astype(np.uint8)
        result = np.ones((gb.shape[0], gb.shape[1], 3))
        result = gb[..., np.newaxis] * result
        peak_i = peak_in % 20
        result[..., 0][result[..., 0] != 0] = 1 * gb[gb != 0]
        result[..., 1][result[..., 1] != 0] = 0 * gb[gb != 0]
        result[..., 2][result[..., 2] != 0] = 1 * gb[gb != 0]
        gbs_coloring.append(result)
        # plt.imshow(result), plt.show()
        cv2.imwrite(
            str(save_path.joinpath("{:04d}.png".format(peak_in))), result
        )


files = sorted(Path(
    "/home/kazuya/file_server2/miccai/all_outputs/guided/00001/each_peak/"
).glob("*.mat"))
output = Path("output_file")
output.mkdir(parents=True, exist_ok=True)
gbs = []
gbs.append(loadmat(f"/home/kazuya/file_server2/miccai/all_outputs/guided/00001/each_peak/0053.mat")["image"])
gbs.append(loadmat(f"/home/kazuya/file_server2/miccai/all_outputs/guided/00001/each_peak/0056.mat")["image"])
coloring(gbs, output)
