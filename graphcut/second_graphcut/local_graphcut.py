from utils import *
import numpy as np
import torch
from first_graphcut import GraphCut


class LocalGraphCut(GraphCut):
    def __init__(self, weight_path = None, plot_size=6):
        super().__init__()
        self.pairwise_weight = 1
        self.dataterm_weight = 1
        self.pairwise_gaussian = 0.001
        self.pairwise_bias = 0
        self.mask = None
        self.peak = None
        self.note_likely_map = None
        self.likely_map = None
        self.note_peak_backprop = None
        self.peak_backprop = None
        if weight_path is None:
            weight_path = f"./weight/challenge/best_{plot_size}.pth"
        net = UNet(n_channels=1, n_classes=1)
        net.load_state_dict(
            torch.load(
                str(weight_path),
                map_location={"cuda:3": "cuda:0"},
            )
        )
        self.gb_model = GuidedBackpropReLUModel(model=net, use_cuda=True)

    def add_t_links(self):
        # data項のedge
        for row_index in range(self.row):
            for column_index in range(self.column):
                current_index = self.row * column_index + row_index

                capacity = self.dataterm_weight * (
                    self.note_peak_backprop[row_index, column_index]
                    + self.note_likely_map[row_index, column_index]
                )
                self.graph.add_edge("s", current_index, capacity=capacity)

                capacity = self.dataterm_weight * (
                    self.peak_backprop[row_index, column_index]
                    + self.likely_map[row_index, column_index]
                )
                self.graph.add_edge(current_index, "t", capacity=capacity)

        # foreground seed
        x, y = np.where(
            (self.note_likely_map > self.note_likely_map.max() * 0.9)
            | (self.note_peak_backprop > self.note_peak_backprop.max() * 0.9)
        )
        # x, y = np.where(self.note_likely_map > 240)
        for index in zip(x, y):
            current_index = self.row * index[1] + index[0]
            self.graph.add_edge("s", current_index, capacity=1_000_000)

        # background seed
        x, y = np.where(
            (self.likely_map > self.likely_map.max() * 0.9)
            | (self.peak_backprop > self.peak_backprop.max() * 0.9)
        )
        # x, y = np.where(self.likely_map > 240)
        for index in zip(x, y):
            current_index = self.row * index[1] + index[0]
            self.graph.add_edge(current_index, "t", capacity=1_000_000)

    def backprop(self, peaks, mask):
        self.mask = mask
        self.peak = peaks[0]
        self.calculate_peak_backprop(peaks)

    def calculate_peak_backprop(self, peaks):
        likelys = []
        for peak in peaks:
            likely = np.zeros(self.mask.shape)
            likely[peak[1], peak[0]] = 255
            likely = gaus_filter(likely, 101, 6)
            likely = 0.5 * likely / likely.max()
            likelys.append(likely)
        self.note_likely_map = likelys.pop()
        self.likely_map = np.max(likelys, axis=0)

        peak_backprops = []
        img = self.original
        img = (img.astype(np.float32) / 255).reshape((1, 1, img.shape[0], img.shape[1]))
        img = torch.from_numpy(img)
        img.requires_grad = True

        # peak response
        for peak in peaks:
            peak_backprop = self.gb_model(img, peak[0], peak[1], 1, index=None)
            peak_backprop = 0.5 * peak_backprop / peak_backprop.max()
            peak_backprop = peak_backprop.clip(0, 255)
            peak_backprops.append(peak_backprop)

        self.note_peak_backprop = peak_backprops.pop()
        self.peak_backprop = np.max(peak_backprops, axis=0)
