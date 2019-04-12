from pathlib import Path
from datetime import datetime
import torch
import os
os.chdir(Path.cwd().parent)
from propagation import BackPropBackGround, BackpropAll, TopDown,  BackPropBacks
from networks import UNet, MargeNet
from propagation import GuideCall, GuideOnly


torch.cuda.set_device(0)
if __name__ == "__main__":
    gpu = True
    norm = True
    plot_size = 9
    radius = 1
    date = datetime.now().date()
    key = 3

    cross = 0
    dataset = 'GBM'
    dirs = sorted(Path("/home/kazuya/file_server2/images/{}_cut/".format(dataset)).iterdir())

    input_path = sorted(Path("/home/kazuya/file_server2/images/sequ18/ori").glob('*.tif'))
    output_path = Path("/home/kazuya/file_server2/all_outputs/guided_only/test".format(dataset))
    # weight_path = "/home/kazuya/file_server2/weights/{}/{}/best_{}.pth".format(dataset, cross, plot_size)
    weight_path = "/home/kazuya/file_server2/weights/marge/best2_12.pth".format(dataset, cross, plot_size)
    net = UNet(n_channels=1, n_classes=1, sig=norm)

    net = MargeNet(n_channels=1, n_classes=1, sig=norm, net=net)
    net.load_state_dict(
        torch.load(weight_path, map_location={"cuda:2": "cuda:0"})
    )

    call = [GuideCall, GuideOnly]
    bp = call[1](input_path, output_path, net, marge=True)
    bp.main()
