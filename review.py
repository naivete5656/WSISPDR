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
from utils import EvaluationMethods


class UseMethods(EvaluationMethods):
    def evaluation_all(self):
        evaluations = []
        for path in zip(self.pred_paths, self.target_path):
            # pred = cv2.imread(str(path[0]), 0)
            pred = np.load(path[0]).astype(np.int)
            target = cv2.imread(str(path[1]), 0)
            evaluations = self.update_evaluation(pred, target, evaluations)
        self.review(evaluations)

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
            output_path = path[0].parent.parent.joinpath("sophisticated_pred")
            output_path.mkdir(parents=True, exist_ok=True)
            np.save(output_path.joinpath(f"{img_i:05d}.npy"), new_pred)


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
    target_path = sorted(Path("/home/kazuya/file_server2/groundTruths/challenge/01_SEG/SEG_cut").glob('*.tif'))
    pred_path = sorted(Path("/home/kazuya/file_server2/all_outputs/bes_out/challenge_01/sophisticated_pred").glob('*.npy'))
    detection_path = sorted(Path("/home/kazuya/file_server2/all_outputs/bes_out/challenge_01/pred").glob('*.tif'))
    save_path = Path("/home/kazuya/file_server2/all_outputs/bes_out/challenge_01/txt_result")

    evaluation = UseMethods(pred_path, target_path,detection_path, save_path=save_path)
    # evaluation.noize_off()
    evaluation.evaluation_all()
