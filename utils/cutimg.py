import cv2
from pathlib import Path
from PIL import Image
import numpy as np


def cut_image(plot_size="6", sequence=18, size=320, override=100, norm_value=255):
    paths = sorted(
        Path(
            "/home/kazuya/weakly_supervised_instance_segmentation/images/review/sparce"
        ).glob("*.png")
    )
    i = 0
    savepath = Path(
        "/home/kazuya/weakly_supervised_instance_segmentation/images/review/sparce-cut"
    )
    savepath.mkdir(parents=True, exist_ok=True)
    for path in paths:
        img = np.array(Image.open(path)).astype(np.float32)
        img = (img / img.max() * 255).astype(np.uint8)
        for x in range(0, img.shape[0] - size, size - override):
            for y in range(0, img.shape[1] - size, size - override):
                cv2.imwrite(
                    str(savepath / Path("%05d.tif" % i)),
                    img[x : x + size, y : y + size],
                )
                i += 1
            cv2.imwrite(
                str(savepath / Path("%05d.tif" % i)),
                img[x : x + size, img.shape[1] - size : img.shape[1]],
            )
            i += 1
        for y in range(0, img.shape[1] - size, size - override):
            cv2.imwrite(
                str(savepath / Path("%05d.tif" % i)),
                img[img.shape[0] - size : img.shape[0], y : y + size],
            )
            i += 1
        cv2.imwrite(
            str(savepath / Path("%05d.tif" % i)),
            img[img.shape[0] - size : img.shape[0], img.shape[1] - size : img.shape[1]],
        )
        i += 1
    print(i)


if __name__ == "__main__":
    cut_image()
