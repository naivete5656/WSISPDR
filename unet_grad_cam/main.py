from call_backprop import *
from pathlib import Path

from datetime import datetime
import torch
from top_down_call import TopDown


if __name__ == "__main__":
    gpu = True
    plot_size = 12
    radius = 1
    date = datetime.now().date()
    key = 2

    input_path = sorted(Path("../images/sequ_cut/sequ18/ori").glob("*.tif"))
    output_path = Path(f"./output/{date}/test")
    weight_path = f"../weights/MSELoss/best_{plot_size}.pth"

    torch.cuda.set_device(0)
    call_method = {
        0: TopDown,
        1: BackpropAll,
        2: BackPropBackGround,
    }
    bp = call_method[key](input_path, output_path, weight_path)
    bp.main()
