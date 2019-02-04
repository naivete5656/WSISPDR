from second_graphcut import LocalGraphCut
import numpy as np
import cv2
from PIL import Image
from utils import local_maxima
import numpy as np
from skimage import measure
import collections
import matplotlib.pyplot as plt
import cv2


def local_graphcut(paths, mask, local_peaks, weight_path):
    segments = []
    peaks = local_peaks
    graph = LocalGraphCut(weight_path=weight_path)
    graph.update_image(paths=paths)
    print("local")
    number_of_roop = local_peaks.shape[0] - 1
    for i in range(number_of_roop):
        graph.backprop(peaks=local_peaks, mask=mask)
        result = graph.graphcut()

        print("finish local graphcut")

        segments.append(result * mask)
        mask = np.clip(mask - result, 0, 1)

        local_peaks = np.delete(local_peaks, local_peaks.shape[0] - 1, axis=0)
    segments.append(mask)

    segments = fix_segment(segments, peaks)

    return segments


def second_graphcut(paths, img, weight_path):
    print("local graphcut")
    img = (img * 255).astype(np.uint8)
    # labeling
    _, label_image = cv2.connectedComponents(img, connectivity=4)

    # peak
    detection_result = np.array(Image.open(paths[1]))[:320, :320]
    peaks = local_maxima(detection_result)

    # segment内のpeakを計算
    counted_labels, peak_label_association = sophisticate_segment(
        img, peaks, label_image
    )

    segmentation_results = np.zeros(img.shape)
    new_labels = 1
    for label_num in counted_labels.keys():
        number_of_peak = counted_labels[label_num]
        if number_of_peak > 1:

            # graphcut mask make
            for_graphcut_mask = np.zeros(img.shape)
            for_graphcut_mask[label_image == label_num] = 1

            # local peak の座標を取得する
            peak_index = peak_label_association[
                peak_label_association[:, 1] == label_num
            ][:, 0]
            local_peaks = peaks[peak_index]

            plt.imshow(for_graphcut_mask),plt.plot(local_peaks[:,0],local_peaks[:,1],'rx'),plt.show()
            segments = local_graphcut(
                paths, for_graphcut_mask, local_peaks, weight_path=weight_path
            )

            for segment in segments:
                segment[segment != 0] = new_labels
                segmentation_results = np.maximum(segmentation_results, segment)
                new_labels += 1
        else:
            segmentation_results[label_image == label_num] = new_labels
            new_labels += 1
    print("finish")
    return segmentation_results


def boundary_recognize(result):
    boundaries = []
    for i in range(1, int(result.max() + 1)):
        temp = np.zeros(result.shape)
        temp[result == i] = result[result == i]
        boundary = np.zeros(result.shape)
        try:
            contours = measure.find_contours(temp, 0.8)
            for contour in contours:
                for plot in contour:
                    boundary[int(plot[0]), int(plot[1])] = i + 1
        except IndexError:
            print("Skip")
        boundaries.append(boundary)
    try:
        boundary = np.max(boundaries, axis=0)
    except ValueError:
        boundary = np.zeros(result.shape)
    return boundary


def fix_segment(segments, peaks):
    # make instances
    label_id = 1
    img = np.zeros(segments[0].shape)
    for segment in segments:
        img[segment > 0] = label_id
        label_id += 1
    label_max = int(img.max())

    for peak in peaks:
        # make mask for each segment
        peak_value = img[peak[1], peak[0]]
        mask = np.zeros(img.shape, dtype=np.uint8)
        mask[img == peak_value] = 255
        n_labels, label_image = cv2.connectedComponents(mask, connectivity=4)
        segment_peak_value = label_image[peak[1], peak[0]]

        for i in range(1, n_labels):
            if i != segment_peak_value:
                mask = np.zeros(img.shape)
                mask[label_image == i] = 1

                labels = []
                # calculate surround value
                contours = measure.find_contours(mask, 0.8)[0]
                for contour in contours:
                    label = img[int(contour[0]), int(contour[1])]
                    if (label != 0) and (label != peak_value):
                        labels.append(label)

                kernel = np.ones((2, 2), np.uint8)

                erosion = cv2.erode(mask, kernel=kernel, iterations=1)
                if erosion.max() > 0:
                    contours = measure.find_contours(erosion, 0.8)[0]
                    for contour in contours:
                        label = img[int(contour[0]), int(contour[1])]
                        if (label != 0) and (label != peak_value):
                            labels.append(label)

                dilation = cv2.dilate(mask, kernel=kernel, iterations=1)

                contours = measure.find_contours(dilation, 0.8)[0]
                for contour in contours:
                    label = img[int(contour[0]), int(contour[1])]
                    if (label != 0) and (label != peak_value):
                        labels.append(label)

                dilation = cv2.dilate(
                    mask, kernel=np.ones((3, 3), np.uint8), iterations=1
                )

                contours = measure.find_contours(dilation, 0.8)[0]
                for contour in contours:
                    label = img[int(contour[0]), int(contour[1])]
                    if (label != 0) and (label != peak_value):
                        labels.append(label)

                counted_label = collections.Counter(labels)
                try:
                    max_key = max(counted_label, key=counted_label.get)
                except ValueError:
                    max_key = peak_value
                img[label_image == i] = max_key

    segments = []
    for i in range(1, label_max + 1):
        segments.append(np.where(img == i, 1, 0))
    return segments


def sophisticate_segment(img, peaks, label_image):
    """
    :param img:
    :param peaks:
    :return:
    """
    # peakがないsegmentを削除
    mask = np.zeros(img.shape)
    label_nums = []
    peak_label_association = []
    for i, coordinate in enumerate(peaks):
        label_num = label_image[coordinate[1], coordinate[0]]
        if label_num != 0:
            mask[label_image == label_num] = i
            label_nums.append(label_num)
            peak_label_association.append([i, label_num])
    count_label = collections.Counter(label_nums)
    return count_label, np.array(peak_label_association)
