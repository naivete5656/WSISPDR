from pathlib import Path
from datetime import datetime
import torch
import os
os.chdir(Path.cwd().parent)
from propagation import GuideCall, GuideOnly
from networks import UNet

datasets = ['GBM', 'B23P17', 'elmer']
torch.cuda.set_device(0)
if __name__ == "__main__":
    gpu = True
    norm = True
    plot_size = 9
    radius = 1
    date = datetime.now().date()
    key = 0
    cross = 4
    dataset = datasets[1]
    dirs = sorted(Path("/home/kazuya/file_server2/images/{}_cut/".format(dataset)).iterdir())

    input_path = dirs.pop(cross).joinpath('ori').glob('*.tif')
    output_path = Path("/home/kazuya/file_server2/all_outputs/gradcam/{}".format(dataset))
    if dataset == 'elmer':
        weight_path = "/home/kazuya/file_server2/weights/{}/{}/best.pth".format(dataset, plot_size)
    else:
        weight_path = "/home/kazuya/file_server2/weights/{}/{}/best_{}.pth".format(dataset, cross, plot_size)
    net = UNet(n_channels=1, n_classes=1, sig=norm)
    net.load_state_dict(
        torch.load(weight_path, map_location={"cuda:2": "cuda:0"})
    )
    call = [GuideCall,GuideOnly]
    bp = call[1](input_path, output_path, net)
    bp.main()
