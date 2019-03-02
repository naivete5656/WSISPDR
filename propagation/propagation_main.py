from pathlib import Path
from datetime import datetime
import torch
import sys

from call_backprop  import BackPropBackGround, BackpropAll
from top_down_call import TopDown


if __name__ == "__main__":
    gpu = True
    norm = True
    plot_size = 12
    radius = 1
    date = datetime.now().date()
    key = 2

    input_path = sorted(Path("../images/C2C12P7/sequ_cut/0303/sequ9/ori").glob("*.tif"))
    output_path = Path(f"../outputs/gradcam/{date}/sequ09")
    # weight_path = f"../weights/server_weights/encoder_sigmoid/best_{plot_size}.pth"
    weight_path = f"../weights/server_weights/MSELoss/{plot_size}/epoch_weight/00015.pth"

    torch.cuda.set_device(1)
    call_method = {0: TopDown, 1: BackpropAll, 2: BackPropBackGround}
    bp = call_method[key](input_path, output_path, weight_path, sig=norm)
    bp.main()
