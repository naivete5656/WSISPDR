import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from utils import (
    local_maxim,
    target_peaks_gen,
    optimum,
    remove_outside_plot,
    show_res,
    gaus_filter,
)
import collections


class EvaluationMethods:
    def __init__(self, pred_path, target_path, save_path, each_save=False):
        self.pred_paths = sorted(pred_path.glob("*.tif"))
        self.target_path = sorted(target_path.glob("*.tif"))
        self.detection_path = sorted(pred_path.joinpath("detection").glob("*.tif"))
        self.save_path = save_path
        self.each_save = False
        save_path.mkdir(parents=True, exist_ok=True)
        self.calculate = None
        self.mode = None
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
                pred_mask = np.zeros(pred.shape)
            # create mask
            target_mask = np.zeros(target.shape)
            target_mask[target == target_label] = 1
            pred_2 = pred_mask
            target_2 = target_mask
            pred_mask = pred_mask.flatten()
            target_mask = target_mask.flatten()

            tp = pred_mask.dot(target_mask)
            fn = pred_mask.sum() - tp
            fp = target_mask.sum() - tp
            tps += tp
            fns += fn
            fps += fp

            iou = (tp / (tp + fp + fn))
            self.iou_list.append(iou)

            dice = (2 * tp) / (2 * tp + fn + fp)
            self.dice_list.append(dice)

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

        return tp, fn, fp

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
        # detection
        evaluations = np.array(evaluations)
        tps_fns_fps = evaluations.sum(axis=0)

        detection_recall = tps_fns_fps[0] / (tps_fns_fps[0] + tps_fns_fps[1])
        detection_precision = tps_fns_fps[0] / (tps_fns_fps[0] + tps_fns_fps[2])
        detection_f_measure = (2 * detection_recall * detection_precision) / (
            detection_recall + detection_precision
        )

        # segmentation
        dice = (
            2 * tps_fns_fps[3] / (2 * tps_fns_fps[3] + tps_fns_fps[4] + tps_fns_fps[5])
        )
        iou = tps_fns_fps[3] / (tps_fns_fps[3] + tps_fns_fps[4] + tps_fns_fps[5])

        # instance segmentation
        instance_dice = np.nanmean(np.array(self.dice_list))
        instance_iou = np.nanmean(np.array(self.iou_list))
        text = f"detection\n precision:{detection_precision}\nrecall:{detection_recall}\nf-measure:{detection_f_measure}\
                                    \nsegmentation\ndice:{dice}\niou:{iou}\n\
                                    \ninstance-segmentation\ninstance_iou:{instance_iou}\ninstance-dice:{instance_dice}\n"
        print(text)
        plt.hist(self.iou_list)
        with open(self.save_path.joinpath(f"result.txt"), mode="w") as f:
            f.write(text)

    def update_evaluation(self, pred, target, evaluations):

        bou_list = []
        max_bou = target.shape[0]
        bou_list.extend(target[0,:])
        bou_list.extend(target[max_bou - 1,:])
        bou_list.extend(target[:,max_bou - 1])
        bou_list.extend(target[:,0])
        np.unique(bou_list)
        for x in bou_list:
            target[target == x] = 0

        bou_list = []
        max_bou = pred.shape[0]
        bou_list.extend(pred[0, :])
        bou_list.extend(pred[max_bou - 1, :])
        bou_list.extend(pred[:, max_bou - 1])
        bou_list.extend(pred[:, 0])
        np.unique(bou_list)
        for x in bou_list:
            pred[pred == x] = 0

        #plt.imshow(pred), plt.show()
        #plt.imshow(target), plt.show()

        #detection_tp, detection_fn, detection_fp = self.f_measure(pred, target)
        seg_tp, seg_fn, seg_fp = self.segmentation_eval(pred, target)
        self.instance_eval(pred, target)
        evaluations.append(
            [
                1,
                1,
                1,
                seg_tp,
                seg_fn,
                seg_fp,
            ]
        )

        return evaluations


class UseMethods(EvaluationMethods):
    def evaluation_all(self):
        evaluations = []
        for path in zip(self.pred_paths, self.target_path):
            pred = cv2.imread(str(path[0]), 0)
            #pred = np.load(path[0]).astype(np.int)
            target = cv2.imread(str(path[1]), 0)
            evaluations = self.update_evaluation(pred, target, evaluations)
        self.review(evaluations)

    def noize_off(self, pred_path):
        for img_i, path in enumerate(
            zip(self.pred_paths, self.target_path, self.detection_path)
        ):
            pred = cv2.imread(str(path[0]), 0)
            label_image = cv2.connectedComponents(pred)[1]

            # get peal
            detection = cv2.imread(str(path[2]), 0)
            plots = local_maxim(detection, 100, 2)

            # only peak segment
            new_pred = np.zeros(pred.shape)
            values = []
            label = 1
            for i, plot in enumerate(plots):
                value = label_image[int(plot[1]), int(plot[0])]
                values.append(value)
                if value != 0:
                    new_pred[label_image == value] = label
                label += 1

            counts = collections.Counter(values)
            values = np.array(values)
            for key in counts.keys():
                if counts[key] > 1 and key != 0:
                    multi_segment_mask = np.zeros(pred.shape)
                    multi_segment_mask[label_image == key] = 1

                    peak_index = np.where(values == key)[0]
                    local_peaks = plots[peak_index]

                    # gather peak value
                    local_masks = []
                    for peak in local_peaks:
                        local_mask = np.zeros(multi_segment_mask.shape)
                        local_mask[int(peak[1]), int(peak[0])] = 255
                        local_masks.append(gaus_filter(local_mask, 201, 9))
                    index_mask = np.array(local_masks).argmax(axis=0)

                    for i in range(index_mask.max()):
                        temp = np.zeros(pred.shape)
                        temp[index_mask == i] = multi_segment_mask[index_mask == i]
                        new_pred[temp > 0] = label
                        label += 1
            output_path = pred_path.joinpath("sophisticated_pred")
            output_path.mkdir(parents=True, exist_ok=True)
            np.save(output_path.joinpath(f"{img_i:05d}.npy"), new_pred)
            cv2.imwrite(str(output_path.joinpath(f"{img_i:05d}.tif")), new_pred)


class LinearReview(UseMethods):
    def center_get(self, pred):
        # しきい値処理
        thresh, bin_img = cv2.threshold(
            pred.astype(np.uint8), 0, 255, cv2.THRESH_BINARY_INV
        )
        # 色を反転
        bin_img = cv2.bitwise_not(bin_img)
        # ノイズ除去
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)

        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        pred = cv2.connectedComponentsWithStats(sure_fg.astype(np.uint8))[3]

        return pred

    def f_measure_center(self, pred, target):
        target_centers = []
        for target_label in range(1, target.max() + 1):
            y, x = np.where(target == target_label)
            if x.shape[0] != 0:
                x = x.sum() / x.shape[0]
                y = y.sum() / y.shape[0]
                target_centers.append([x, y])
        target_centers = np.array(target_centers)

        pred_centers = self.center_get(pred)
        return pred_centers, target_centers

    def evaluate(self):
        for path in zip(self.pred_paths, self.target_path):
            pred = cv2.imread(str(path[0]))
            gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)

            target = cv2.imread(str(path[1]), 0)

            # しきい値処理
            thresh, bin_img = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            # 色を反転
            bin_img = cv2.bitwise_not(bin_img)
            # ノイズ除去
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)

            # sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)

            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(
                dist_transform, 0.7 * dist_transform.max(), 255, 0
            )
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)

            # Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)

            # Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1

            # Now, mark the region of unknown with zero
            markers[unknown == 255] = 0

            markers = cv2.watershed(pred, markers)

            plt.imshow(markers), plt.show()

    def evaluation_all(self):
        evaluations = []
        for path in zip(self.pred_paths, self.target_path):
            pred = cv2.imread(str(path[0]), 0)
            pred = cv2.connectedComponents(pred)[1]
            target = cv2.imread(str(path[1]), 0)
            evaluations = self.update_evaluation(pred, target, evaluations)
        self.review(evaluations)
        self.review(evaluations)


if __name__ == "__main__":
    date = datetime.now().date()
    target_path = Path("./images/review/target-cut")
    pred_path = Path("./outputs/labelresults")
    save_path = Path(f"./outputs/txt_result/final")

    evaluation = UseMethods(pred_path, target_path, save_path=save_path)

    # evaluation.noize_off(pred_path)

    #pred_path = pred_path.joinpath("sophisticated_pred")
    # evaluation.pred_paths = sorted(pred_path.glob("*.npy"))
    evaluation.evaluation_all()
    # evaluation.evaluation_iou()
    #
    # target_path = Path("./images/review/target-cut")
    # pred_path = Path("./images/review/sparce-cut")
    # save_path = Path(f"./outputs/txt_result/sparce-cut")
    # eval = LinearReview(pred_path, target_path, save_path=save_path)
    # eval.evaluation_all()
