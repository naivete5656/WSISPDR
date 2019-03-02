from detection_train import TrainNet


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
        val_path,
        val_mask_path,
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
            val_path,
            weight_path,
            save_path,
            norm,
        )
        self.ori_path = train_path
        self.mask_path = mask_path
        self.val_path = val_path
        self.val_mask_path = val_mask_path
