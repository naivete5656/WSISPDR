from pathlib import Path
from datetime import datetime
import torch
import os

os.chdir(Path.cwd().parent)
from propagation import GuideCall, GuideOnly
from networks import UNet, MargeNet

datasets = {1: "GBM", 2: "B23P17", 3: "elmer", 4: "challenge", 5: "0318_9",6: "sequence_10"}
torch.cuda.set_device(0)
if __name__ == "__main__":
    gpu = True
    norm = True
    plot_size = 9
    radius = 1
    date = datetime.now().date()
    key = 0
    cross = 0
    dataset = datasets[6]
    # dirs = sorted(
    #     Path("/home/kazuya/file_server2/images/{}_cut/".format(dataset)).iterdir()
    # )
    #
    # input_path = dirs.pop(cross).joinpath("ori").glob("*.tif")
    input_path = sorted(Path("/home/kazuya/file_server2/images/elmer_cut/0/ori").glob(
        "*.tif"
    ))
    output_path = Path("/home/kazuya/file_server2/all_outputs/gradcam/elmer/0".format(dataset))
    # output_path = Path(f"/home/kazuya/file_server2/all_outputs/gradcam/{dataset}/{cross}")
    weight_path = "/home/kazuya/file_server2/weights/elmer/6/best.pth"
    # weight_path = f"/home/kazuya/file_server2/weights/C2C12/0/best_{plot_size}.pth"

    net = UNet(n_channels=1, n_classes=1, sig=norm)

    net.load_state_dict(torch.load(weight_path, map_location={"cuda:2": "cuda:0"}))
    # weight_path = "/home/kazuya/file_server2/weights/marge/best_12.pth"
    # net = UNet(n_channels=1, n_classes=1, sig=norm)
    # net = MargeNet(n_channels=1, n_classes=1, sig=norm, net=net)
    # net.cuda()
    # net.load_state_dict(torch.load(weight_path, map_location={"cuda:3": "cuda:1"}))
    call = [GuideCall, GuideOnly]
    bp = call[0](input_path, output_path, net)
    bp.main()
