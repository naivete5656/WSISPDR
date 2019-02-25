
from pathlib import Path
from unet_grad_cam import *
from datetime import datetime
import torch


if __name__ == "__main__":
    gpu = True
    plot_size = 12
    radius = 1
    date = datetime.now().date()
    key = 1

    input_path = sorted(Path("./images/sequ_cut/sequ18/12").glob("*.tif"))
    output_path = Path(f"./outputs/gradcam/{date}/encoder")
    weight_path = f"./weights/server_weights/encoder_sigmoid/best_{plot_size}.pth"
    # weight_path = f"./weights/MSELoss/best_{plot_size}.pth"

    torch.cuda.set_device(1)
    call_method = {
        0: TopDown,
        1: BackpropAll,
        2: BackPropBackGround,
    }
    bp = call_method[key](input_path, output_path, weight_path, sig=False)
    bp.main()
