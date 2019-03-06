
from .detection_train import TrainNet
from .custom_loss import SignMseLoss


class TrainNetSecond(TrainNet):
    def __init__(
        self,
        net,
        mode,
        epochs,
        batch_size,
        lr,
        gpu,
        plot_size,
        train_path,
        mask_path,
        weight_path,
        save_path,
        norm,
    ):
        super().__init__(
            net,
            mode,
            epochs,
            batch_size,
            lr,
            gpu,
            plot_size,
            train_path,
            weight_path,
            save_path,
            norm,
        )
        self.ori_path = train_path
        self.mask_path = mask_path
        # self.val_path = val_path
        # self.val_mask_path = val_mask_path
        self.criterion2 = SignMseLoss()

    def loss_calcurate(self, masks_probs_flat, true_masks_flat):
        loss1 = self.criterion(masks_probs_flat, true_masks_flat)
        loss2 = self.criterion2(masks_probs_flat, true_masks_flat)
        return loss1 + loss2
