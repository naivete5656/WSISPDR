from PIL import Image
from unet import *
from pathlib import Path
from gradcam import *
from utils import *
from datetime import datetime
import torch
import numpy as np
import cv2
from skimage.feature import peak_local_max
from torch.autograd import Function
from torchvision import utils
import matplotlib.pyplot as plt
import random
from call_backprop import *


if __name__ == "__main__":
    gpu = True
    plot_size = 12
    radius = 1
    date = datetime.now().date()

    # input_path = sorted(Path("../images/sequence/cuts/test18").glob("*.tif"))
    # output_path = Path(f"./output/{date}/Top-down")
    # weight_path = f"../weights/MSELoss/best_{plot_size}.pth"
    #
    # input_path = sorted(Path("../images/challenge/challenge2").glob("*.tif"))
    # output_path = Path(f"./output/{date}/challenge")
    # weight_path = f"../weights/challenge/best_{plot_size}.pth"
    input_path = sorted(Path("../images/sequ_cut/sequ18/ori").glob("*.tif"))
    output_path = Path(f"./output/{date}/test18")
    weight_path = f"../weights/MSELoss/best_{plot_size}.pth"

    torch.cuda.set_device(0)

    # net = UNet(n_channels=1, n_classes=1)
    # net.load_state_dict(torch.load(weight_path, map_location={"cuda:3": "cuda:0"}))
    # response_map = TopDownBackprop(net)
    # response_map.inference()

    # bp = TopDown(input_path, output_path, weight_path)
    # bp.main()

    # bp = BackPropBackGround(GuidedBackpropReLUModel
    #     input_path=input_path, output_path=output_path, weight_path=weight_path
    # )
    #
    # bp.main()

    bp = BackpropagationEachPeak(
        input_path=input_path, output_path=output_path, weight_path=weight_path
    )

    bp.main()

    # bp = BackpropAll(
    #     input_path=input_path, output_path=output_path, weight_path=weight_path
    # )
    #
    # bp.main()



