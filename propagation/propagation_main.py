from pathlib import Path
from datetime import datetime
import torch
import os
os.chdir(Path.cwd().parent)
from propagation import BackPropBackGround, BackpropAll, TopDown

torch.cuda.set_device(0)
if __name__ == "__main__":
    gpu = True
    norm = True
    plot_size = 12
    radius = 1
    date = datetime.now().date()
    key = 2

    input_path = sorted(Path("./images/elmer_cut/heavy5/ori").glob("*.tif"))
    output_path = Path("/home/kazuya/file_server2/outputs/gradcam/sequ9")
    # weight_path = f"./weights/server_weights/encoder_sigmoid/best_{plot_size}.pth"
    weight_path = (
        "./weights/best.pth"
    )

    call_method = {0: TopDown, 1: BackpropAll, 2: BackPropBackGround}
    bp = call_method[key](input_path, output_path, weight_path)
    bp.main()
