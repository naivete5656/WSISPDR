import cv2
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json
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
from statistics import mean


def original_add_pred():
    # original 画像に　外線を加える
    img = cv2.imread("./outputs/2019-01-18/test/mask_pred/00009.tif", 0)
    # img = cv2.imread("./outputs/2019-01-18/test/mask_pred/00000.tif", 0)
    mask = np.zeros(img.shape, dtype=np.uint8)
    plt.imshow(img > 0.5), plt.show()
    contours = measure.find_contours(img, 0.5)
    for contour in contours:
        for x, y in contour:
            mask[int(x), int(y)] = 255
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # mask_color = np.zeros(img.shape)
    # mask_color[:, :, 0] = mask
    original = cv2.imread("./images/test/exp1_F0003-00700.tif", -1)
    # original = cv2.imread("./images/test/exp1_F0002-00300.tif", -1)
    original = (original / 4096 * 255).astype(np.uint8)
    original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    mask_color = np.zeros(original.shape, dtype=np.uint8)
    mask_color[:, :, 2] = mask
    img = cv2.addWeighted(original, 0.8, mask_color, 0.2, 1)
    cv2.imwrite("out.png", img)


def make_ground_truth(file_path, save_path):
    with open(file_path) as f:
        df = json.load(f)
        im = Image.new("I", (1392, 1040), 0)
        draw = ImageDraw.Draw(im)
        for i, label in enumerate(df["shapes"]):
            plots = []
            for x, y in label["points"]:
                plots.append((x, y))
            draw.polygon(tuple(plots), fill=i + 1)
        mask = np.array(im)
        cv2.imwrite(save_path, mask)


def calculate_dice(pred, target):
    pred = pred.flatten()
    target = target.flatten()
    numerator = pred.dot(target)
    denominator = pred.sum() + target.sum()
    t = 2 * numerator / denominator
    return t


def calculate_iou(pred, target):
    pred = pred.flatten()
    target = target.flatten()
    numerator = pred.dot(target)
    denominator = pred.sum() + target.sum() - pred.dot(target)
    t = numerator / denominator
    return t


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
        self.methods = [calculate_dice, calculate_iou]

    def save_result(self, mode, result):
        save_path = self.save_path / Path(self.mode + f"{mode}.txt")
        text = f"{mode} {self.mode} result =  {result}"
        print(text)
        with open(save_path, mode="w") as f:
            f.write(text)

    def instance_eval(self, pred, target):
        assert pred.shape == target.shape, print("different shape at pred and target")
        tp = 0
        fp = 0
        fn = 0
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

            pred = pred_mask.flatten()
            target = target_mask.flatten()
            tp += pred.dot(target)
            fn += pred.sum() - tp
            fp += target.sum() - tp
        return tp, fn, fp

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

        # nlabels, labelimage = cv2.connectedComponents(pred)
        # pred_centers = cv2.connectedComponentsWithStats(pred)[3]

        associate_id = optimum(target_centers, pred_centers, 40)

        target_final, no_detected_id = remove_outside_plot(
            target_centers, associate_id, 0, target.shape
        )
        pred_final, overdetection_id = remove_outside_plot(
            pred_centers, associate_id, 1, target.shape
        )
        show_res(target, target_centers, pred_centers, no_detected_id, overdetection_id)

        tp = associate_id.shape[0]
        fn = target_final.shape[0] - associate_id.shape[0]
        fp = pred_final.shape[0] - associate_id.shape[0]
        return tp, fn, fp


class UseMethods(EvaluationMethods):

    def evaluation_all(self):
        evaluations_detection = []
        evaluations_segmentation = []
        evaluations_instance = []
        for path in zip(self.pred_paths, self.target_path):
            pred = cv2.imread(str(path[0]), 0)
            _, pred = cv2.connectedComponents(pred)
            target = cv2.imread(str(path[1]), 0)
            evaluations_detection.append(self.f_measure(pred, target))
            evaluations_segmentation.append(self.segmentation_eval(pred, target))
            evaluations_instance.append(self.instance_eval(pred, target))

        # detection
        evaluations = np.array(evaluations_detection)
        tps = np.sum(evaluations[:, 0])
        fns = np.sum(evaluations[:, 1])
        fps = np.sum(evaluations[:, 2])

        detection_recall = tps / (tps + fns)
        detection_precision = tps / (tps + fps)
        detection_f_measure = (2 * detection_recall * detection_precision) / (
            detection_recall + detection_precision
        )

        # segmentation
        evaluations = np.array(evaluations_segmentation)
        tps = np.sum(evaluations[:, 0])
        fns = np.sum(evaluations[:, 1])
        fps = np.sum(evaluations[:, 2])

        segmentation_recall = tps / (tps + fns)
        segmentation_precision = tps / (tps + fps)
        segmentation_f_measure = (2 * segmentation_recall * segmentation_precision) / (
            segmentation_recall + segmentation_precision
        )
        dice = 2 * tps / (2 * tps + fps + fns)
        iou = tps / (tps + fns + fps)

        # instance segmentation
        evaluations = np.array(evaluations_segmentation)
        tps = np.sum(evaluations[:, 0])
        fns = np.sum(evaluations[:, 1])
        fps = np.sum(evaluations[:, 2])

        instance_recall = tps / (tps + fns)
        instance_precision = tps / (tps + fps)
        instance_f_measure = (2 * instance_recall * instance_precision) / (
            instance_recall + instance_precision
        )
        instance_dice = 2 * tps / (2 * tps + fps + fns)
        instance_iou = tps / (tps + fns + fps)

        text = f"precision:{detection_precision}\nrecall:{detection_recall}\nf-measure:{detection_f_measure}\n\
            segmentation_precision:{segmentation_precision}\n segmentation_recall:{segmentation_recall}\n segmentation_f-measure:{segmentation_f_measure}\n\
            instance_precision:{instance_precision}\ninstance_recall:{instance_recall}\n instance_f-measure:{instance_f_measure}\niou:{iou}\n\
            instance_iou:{instance_iou}\ndice:{dice}\ninstance-dice:{instance_dice}\n"
        print(text)
        with open(self.save_path.joinpath(f"result.txt"), mode="w") as f:
            f.write(text)

    def evaluation_iou(self):
        for path in zip(self.pred_paths, self.target_path):
            pred = cv2.imread(str(path[0]), 0)
            target = cv2.imread(str(path[1]), 0)
            self.calculate = calculate_iou
            self.instance_eval(pred, target)

    def noize_off(self):
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
                    # plt.imshow(new_pred), plt.show()
            plt.imshow(new_pred), plt.show()
            cv2.imwrite(str(f"./outputs/pred/sophisticated_pred/{img_i}.tif"), new_pred)


class LinearReview(UseMethods):
    def center_get(self, pred):
        # しきい値処理
        thresh, bin_img = cv2.threshold(
            pred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
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
            dist_transform, 0.5 * dist_transform.max(), 255, 0
        )
        pred = cv2.connectedComponentsWithStats(sure_fg.astype(np.uint8))[3]

        return pred

    def f_measure_center(self, pred, target):
        target_centers = []
        for target_label in range(1, target.max() + 1):
            y, x = np.where(target == target_label)
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
        evaluations_detection = []
        evaluations_segmentation = []
        evaluations_instance = []
        for path in zip(self.pred_paths, self.target_path):
            pred = cv2.imread(str(path[0]), 0)
            ret, pred = cv2.threshold(pred, 0, 125, cv2.THRESH_BINARY)

            target = cv2.imread(str(path[1]), 0)
            evaluations_detection.append(self.f_measure(pred, target))
            evaluations_segmentation.append(self.segmentation_eval(pred, target))
            evaluations_instance.append(self.instance_eval(pred, target))

        # detection
        evaluations = np.array(evaluations_detection)
        tps = np.sum(evaluations[:, 0])
        fns = np.sum(evaluations[:, 1])
        fps = np.sum(evaluations[:, 2])

        detection_recall = tps / (tps + fns)
        detection_precision = tps / (tps + fps)
        detection_f_measure = (2 * detection_recall * detection_precision) / (
            detection_recall + detection_precision
        )

        # segmentation
        evaluations = np.array(evaluations_segmentation)
        tps = np.sum(evaluations[:, 0])
        fns = np.sum(evaluations[:, 1])
        fps = np.sum(evaluations[:, 2])

        segmentation_recall = tps / (tps + fns)
        segmentation_precision = tps / (tps + fps)
        segmentation_f_measure = (2 * segmentation_recall * segmentation_precision) / (
            segmentation_recall + segmentation_precision
        )
        dice = 2 * tps / (2 * tps + fps + fns)
        iou = tps / (tps + fns + fps)

        # instance segmentation
        evaluations = np.array(evaluations_segmentation)
        tps = np.sum(evaluations[:, 0])
        fns = np.sum(evaluations[:, 1])
        fps = np.sum(evaluations[:, 2])

        instance_recall = tps / (tps + fns)
        instance_precision = tps / (tps + fps)
        instance_f_measure = (2 * instance_recall * instance_precision) / (
            instance_recall + instance_precision
        )
        instance_dice = 2 * tps / (2 * tps + fps + fns)
        instance_iou = tps / (tps + fns + fps)

        text = f"precision:{detection_precision}\nrecall:{detection_recall}\nf-measure:{detection_f_measure}\n\
                    segmentation_precision:{segmentation_precision}\n segmentation_recall:{segmentation_recall}\n segmentation_f-measure:{segmentation_f_measure}\n\
                    instance_precision:{instance_precision}\ninstance_recall:{instance_recall}\n instance_f-measure:{instance_f_measure}\niou:{iou}\n\
                    instance_iou:{instance_iou}\ndice:{dice}\ninstance-dice:{instance_dice}\n"
        print(text)
        with open(self.save_path.joinpath(f"result.txt"), mode="w") as f:
            f.write(text)


if __name__ == "__main__":
    date = datetime.now().date()
    # target_path = Path("./outputs/target")
    # pred_path = Path("./outputs/pred/sophisticated_pred")
    # save_path = Path(f"./outputs/{date}/quantitive")
    # evaluation = UseMethods(pred_path, target_path, save_path=save_path)
    # evaluation.noize_off()
    # evaluation.evaluation_all()
    # evaluation.evaluation_iou()
    # original_add_pred()

    target_path = Path("./outputs/target")
    pred_path = Path("./images/precond_linear")
    save_path = Path(f"./outputs/{date}/linear")
    eval = LinearReview(pred_path, target_path, save_path=save_path)
    eval.evaluation_all()
