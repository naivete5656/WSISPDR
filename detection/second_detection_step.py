import torch
from datetime import datetime
from pathlib import Path
import os
from pathlib import Path
os.chdir(Path.cwd().parent)
from detection import TrainNetSecond
from networks import UNet, UnetMultiFixedWeight


torch.cuda.set_device(1)
mode = "single"
plot_size = 12
norm = False
date = datetime.now().date()
# train_path = [Path("./images/sequ_cut/sequ9"), Path("./images/sequ_cut/sequ17")]
train_path = Path('/home/kazuya/main/weakly_supervised_instance_segmentation/outputs/sequ18_cut/2019-03-02/pred')
mask_path = train_path.parent.joinpath('gt')
# val_path = Path('/home/kazuya/main/weakly_supervised_instance_segmentation/outputs/sequ18_cut/2019-03-02/pred')
# val_mask_path = train_path.parent.joinpath('gt')
save_weight_path = Path("./weights/{}/second_net/{}/best.pth".format(date, plot_size))
save_path = Path("./confirm")

models = {"single": UNet, "multi": UnetMultiFixedWeight}
net = models[mode](n_channels=1, n_classes=1, sig=norm)
net.cuda()

train = TrainNetSecond(
    net=net,
    mode=mode,
    epochs=500,
    batch_size=1,
    lr=1e-5,
    gpu=True,
    plot_size=plot_size,
    train_path=train_path,
    mask_path=mask_path,
    weight_path=save_weight_path,
    save_path=save_path,
    norm=norm,
)

train.main()
