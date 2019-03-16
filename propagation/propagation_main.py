from pathlib import Path
from datetime import datetime
import torch
import os
os.chdir(Path.cwd().parent)
from propagation import BackPropBackGround, BackpropAll, TopDown,  BackPropBacks
from networks import UNet

torch.cuda.set_device(0)
if __name__ == "__main__":
    gpu = True
    norm = True
    plot_size = 12
    radius = 1
    date = datetime.now().date()
    key = 2

    # input_path = sorted(Path("./images/elmer_cut/heavy5/ori").glob("*.tif"))
    input_path = sorted(Path("./images/C2C12P7/sequ_cut/0303/sequ9/ori").glob("*.tif"))
    # output_path = Path("/home/kazuya/file_server2/outputs/gradcam/heavy")
    output_path = Path("/home/kazuya/file_server2/all_outputs/gradcm/sequ9")
    # weight_path = f"./weights/server_weights/encoder_sigmoid/best_{plot_size}.pth"
    # weight_path = (
    #     "/home/kazuya/file_server2/weights/2019-03-12/challenge/{}/best.pth".format(plot_size)
    # )
    # weight_path = '/home/kazuya/file_server2/weights/2019-03-12/c2c12p7_d_c/12/best.pth'
    weight_path = '/home/kazuya/file_server2/weights/MSELoss/best_12.pth'
    net = UNet(n_channels=1, n_classes=1, sig=norm)
    net.load_state_dict(
        torch.load(weight_path, map_location={"cuda:3": "cuda:0"})
    )

    call_method = {0: TopDown, 1: BackpropAll, 2: BackPropBackGround, 3: BackPropBacks}
    bp = call_method[key](input_path, output_path, net)
    bp.main()
