import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .matching import remove_outside_plot, optimum


class EvaluationMethods:
    def __init__(self, pred_path, target_path, detection_path, save_path, each_save=False):
        self.pred_paths = pred_path
        self.target_path = target_path
        self.detection_path = detection_path
        self.save_path = save_path
        self.each_save = False
        save_path.mkdir(parents=True, exist_ok=True)
        self.calculate = None
        self.mode = None
        self.instance_iou_list = []
        self.instance_dice_list = []
        self.iou_list = []
        self.dice_list = []

    def save_result(self, mode, result):
        save_path = self.save_path / Path(self.mode + f"{mode}.txt")
        text = f"{mode} {self.mode} result =  {result}"
        print(text)
        with open(save_path, mode="w") as f:
            f.write(text)

    def instance_eval(self, pred, target):
        assert pred.shape == target.shape, print("different shape at pred and target")
        tps = 0
        fps = 0
        fns = 0
        for target_label in range(1, target.max() + 1):
            # seek pred label correspond to the label of target
            correspond_labels = pred[target == target_label]
            correspond_labels = correspond_labels[correspond_labels != 0]
            unique, counts = np.unique(correspond_labels, return_counts=True)
            try:
                max_label = unique[counts.argmax()]
                pred_mask = np.zeros(pred.shape)
                pred_mask[pred == max_label] = 1
            except ValueError:
                bou_list = []
                max_bou = target.shape[0]
                max_bou_h = target.shape[1]
                bou_list.extend(target[0, :])
                bou_list.extend(target[max_bou - 1, :])
                bou_list.extend(target[:, max_bou_h - 1])
                bou_list.extend(target[:, 0])
                bou_list = np.unique(bou_list)
                for x in bou_list:
                    target[target == x] = 0

                pred_mask = np.zeros(pred.shape)
            # create mask
            target_mask = np.zeros(target.shape)
            target_mask[target == target_label] = 1
            # plt.imshow(target_mask), plt.show()
            # plt.imshow(pred_mask), plt.show()
            pred_mask = pred_mask.flatten()
            target_mask = target_mask.flatten()

            tp = pred_mask.dot(target_mask)
            fn = pred_mask.sum() - tp
            fp = target_mask.sum() - tp
            tps += tp
            fns += fn
            fps += fp

            iou = (tp / (tp + fp + fn))
            print(iou)
            self.instance_iou_list.append(iou)

            dice = (2 * tp) / (2 * tp + fn + fp)
            self.instance_dice_list.append(dice)

    def segmentation_eval(self, pred, target):
        pred_mask = np.zeros(pred.shape)
        pred_mask[pred > 0] = 1

        target_mask = np.zeros(target.shape)
        target_mask[target > 0] = 1

        pred = pred_mask.flatten()
        target = target_mask.flatten()

        tp = pred.dot(target)
        fn = pred.sum() - tp
        fp = target.sum() - tp
        iou = (tp / (tp + fp + fn))
        self.iou_list.append(iou)

        dice = (2 * tp) / (2 * tp + fn + fp)
        self.dice_list.append(dice)

    def f_measure_center(self, pred, target):
        target_centers = []
        for target_label in range(1, target.max() + 1):
            y, x = np.where(target == target_label)
            if x.shape[0] != 0:
                x = x.sum() / x.shape[0]
                y = y.sum() / y.shape[0]
                target_centers.append([x, y])
        target_centers = np.array(target_centers)

        pred_centers = []
        for pred_label in range(1, pred.max() + 1):
            y, x = np.where(pred == pred_label)
            if x.shape[0] != 0:
                x = x.sum() / x.shape[0]
                y = y.sum() / y.shape[0]
                pred_centers.append([x, y])
        pred_centers = np.array(pred_centers)
        return pred_centers, target_centers

    def f_measure(self, pred, target):
        pred_centers, target_centers = self.f_measure_center(pred, target)
        if pred_centers.shape[0] != 0:
            associate_id = optimum(target_centers, pred_centers, 40)

            target_final, no_detected_id = remove_outside_plot(
                target_centers, associate_id, 0, target.shape
            )
            pred_final, overdetection_id = remove_outside_plot(
                pred_centers, associate_id, 1, target.shape
            )
            # show_res(target, target_centers, pred_centers, no_detected_id, overdetection_id)

            tp = associate_id.shape[0]
            fn = target_final.shape[0] - associate_id.shape[0]
            fp = pred_final.shape[0] - associate_id.shape[0]
        else:
            tp = 0
            fn = 0
            fp = 0
        return tp, fn, fp

    def review(self, evaluations):

        # segmentation
        dice = np.nanmean(np.array(self.dice_list))
        iou = np.nanmean(np.array(self.dice_list))

        # instance segmentation
        instance_dice = np.nanmean(np.array(self.instance_dice_list))
        instance_iou = np.nanmean(np.array(self.instance_iou_list))
        text = "segmentation\ndice:{}\niou:{}\n\
                                    \ninstance-segmentation\ninstance_dice:{}\ninstance-iou:{}\n".format(dice, iou, instance_dice, instance_iou)
        print(text)
        plt.hist(self.instance_iou_list), plt.show()
        with open(self.save_path.joinpath(f"result.txt"), mode="w") as f:
            f.write(text)

    def update_evaluation(self, pred, target, evaluations):
        self.segmentation_eval(pred, target)
        self.instance_eval(pred, target)
