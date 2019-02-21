import torch
from datetime import datetime
from pathlib import Path
from detection import TrainNet
from networks import UNet, UnetMultiFixedWeight


torch.cuda.set_device(1)
mode = "single"
# plot_size = 'gaus'
plot_size = 3
date = datetime.now().date()
# train_path = [Path("./images/sequ_cut/sequ9"), Path("./images/sequ_cut/sequ17")]
train_path = Path("./images/sequ_cut/sequ17")
val_path = Path("./images/sequ_cut/sequ18")
# val_path = Path("./images/sequence/sequ16")
save_weight_path = Path("./weights/{}/normchange/best_{}.pth".format(date, plot_size))
save_path = Path("./confirm")

models = {"single": UNet, "multi": UnetMultiFixedWeight}
net = models[mode](n_channels=1, n_classes=1)
net.cuda()

train = TrainNet(
    net=net,
    mode=mode,
    epochs=500,
    batch_size=9,
    lr=1e-5,
    gpu=True,
    plot_size=plot_size,
    train_path=train_path,
    val_path=val_path,
    weight_path=save_weight_path,
    save_path=save_path,
)

train.main()
