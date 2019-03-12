import torch
from datetime import datetime
from pathlib import Path

if __name__ == "__main__":
    import os

    os.chdir(Path.cwd().parent)
from detection.detection_train import TrainNet
from networks import UNet, UnetMultiFixedWeight, DilatedUNet


torch.cuda.set_device(0)
# mode = "single"
mode = "dilated"
plot_size = 3
norm = True
date = datetime.now().date()
# train_path = [Path("./images/sequ_cut/sequ9"), Path("./images/sequ_cut/sequ17")]
train_path = [
    Path("./images/C2C12P7/sequ_cut/0303/sequ9"),
    Path("./images/C2C12P7/sequ_cut/0318/sequ17"),
]
# train_path = Path("./images/challenge/cut/frame1")
val_path = Path("./images/C2C12P7/sequ_cut/val")
save_weight_path = Path(
    "../weights/test/c2c12p7_d_c/{}/best.pth".format(date, plot_size)
)
save_path = Path("./detection/confirm")
# pre_trained_path = ''
models = {"single": UNet, "multi": UnetMultiFixedWeight, "dilated": DilatedUNet}
net = models[mode](n_channels=1, n_classes=1, sig=norm)
# if pre_trained_path is not None:
#     net.load_state_dict(torch.load(pre_trained_path, map_location={"cuda:3": "cuda:1"}))

net.cuda()

train = TrainNet(
    net=net,
    mode=mode,
    epochs=500,
    batch_size=1,
    lr=1e-5,
    gpu=True,
    plot_size=plot_size,
    train_path=train_path,
    val_path=val_path,
    weight_path=save_weight_path,
    save_path=save_path,
    norm=norm,
)

train.main()
