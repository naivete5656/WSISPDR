
from pathlib import Path
from unet_grad_cam import *
from datetime import datetime
import torch


if __name__ == "__main__":
    gpu = True
    plot_size = 9
    radius = 1
    date = datetime.now().date()
    key = 0

    input_path = sorted(Path("./images/sequ_cut/sequ18/ori").glob("*.tif"))
    output_path = Path(f"./outputs/gradcam/{date}/norm9-top-down")
    weight_path = f"./weights/server_weights/normchange/best_{plot_size}.pth"

    torch.cuda.set_device(0)
    call_method = {
        0: TopDown,
        1: BackpropAll,
        2: BackPropBackGround,
    }
    bp = call_method[key](input_path, output_path, weight_path)
    bp.main()
