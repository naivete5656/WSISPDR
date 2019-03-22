from second_graphcut import second_graphcut, boundary_recognize
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

im_paths = sorted(Path(
    "/home/kazuya/file_server2/images/sequ9/ori"
).glob('*.tif'))
instance_paths = Path(
    "/home/kazuya/file_server2/all_outputs/graphcut/sequ9/labelresults"
)
save_path_seg = Path("/home/kazuya/file_server2/images/for_besnet/sequ9/segmentation")
save_path_seg.mkdir(parents=True, exist_ok=True)
save_path_bou = save_path_seg.parent.joinpath("boundary")
save_path_bou.mkdir(parents=True, exist_ok=True)

save_path_ori = save_path_seg.parent.joinpath("ori")
save_path_ori.mkdir(parents=True, exist_ok=True)


for i, path in enumerate(im_paths):
    ori = cv2.imread(str(path), 0)
    cv2.imwrite(str(save_path_ori.joinpath(f"{i:05d}.tif")), ori)

    shape = ori.shape
    try:
        result = cv2.imread(str(instance_paths.joinpath(f'{i:05d}label.tif')), 0)


        # calculate boundary
        bound = boundary_recognize(result)

        boundary = np.zeros(bound.shape)
        boundary[bound > 0] = 255
        cv2.imwrite(str(save_path_bou.joinpath("{:05d}.tif".format(i))), boundary)
        seg = np.zeros(result.shape)
        seg[result > 0] = 255
        seg[bound > 0] = 0
        cv2.imwrite(str(save_path_seg.joinpath(f"{i:05d}.tif")), seg)
    except AttributeError:
        seg = np.zeros(shape)
        cv2.imwrite(str(save_path_seg.joinpath(f"{i:05d}.tif")), seg)
        boundary = np.zeros(shape)
        cv2.imwrite(str(save_path_bou.joinpath("{:05d}.tif".format(i))), boundary)

    print(i)