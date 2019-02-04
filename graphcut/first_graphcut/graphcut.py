import cv2
import networkx as nx
from scipy.misc import imread
import numpy as np
import math
import matplotlib.pyplot as plt


class _GraphcutParams:
    def __init__(self, dataterm_weight=1, pairwise_weight=10, k=10, bias=0):
        # parameter
        self.pairwise_weight = pairwise_weight
        self.pairwise_gaussian = k
        self.dataterm_weight = dataterm_weight
        self.pairwise_bias = bias
        self.eps = 0.0001

        self.graph = None

        self.original = None
        self.detection_result = None
        self.backprop_result = None
        self.phase_off = None

        self.dataterm = None

        self.row = None
        self.column = None

    def update_image(self, paths):
        self.original = imread(paths[0])
        assert len(self.original.shape) < 3, "次元数が合いません"
        self.row = self.original.shape[0]
        self.column = self.original.shape[1]

        self.detection_result = imread(paths[1])[: self.row, : self.column]
        assert len(self.detection_result.shape) < 3, "次元数が合いません"

        self.backprop_result = imread(paths[2])[: self.row, : self.column]
        assert len(self.backprop_result.shape) < 3, "次元数が合いません"

        self.phase_off = imread(paths[3])[: self.row, : self.column]
        assert len(self.phase_off.shape) < 3, "次元数が合いません"

        self.dataterm_gen()

    def dataterm_gen(self):
        # 大津の閾値判定法
        thresh, _ = cv2.threshold(
            self.phase_off, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # define data term normalize
        dataterm = np.zeros(self.phase_off.shape)
        dataterm[self.phase_off < thresh] = (
            self.phase_off[self.phase_off < thresh] / thresh * 125
        )
        dataterm[self.phase_off > thresh] = (
            self.phase_off[self.phase_off > thresh] - thresh / (255 - thresh)
        ) + 125
        self.dataterm = dataterm


class _NLink(_GraphcutParams):
    def __init__(self):
        super().__init__()

    def column_n_links(self):
        for row_index in range(self.row):
            for column_index in range(self.column - 1):
                capacity = (
                    self.pairwise_weight
                    * math.exp(
                        (-self.pairwise_gaussian)
                        * (
                            self.phase_off[row_index, column_index]
                            - self.phase_off[row_index, column_index + 1]
                        )
                        ** 2
                    )
                    + self.pairwise_bias
                )

                current_index = self.row * column_index + row_index
                self.graph.add_edge(
                    current_index, self.row + current_index, capacity=int(capacity)
                )
                self.graph.add_edge(
                    self.row + current_index, current_index, capacity=int(capacity)
                )

    def row_n_links(self):
        for row_index in range(self.row - 1):
            for column_index in range(self.column):
                capacity = (
                    self.pairwise_weight
                    * math.exp(
                        (-self.pairwise_gaussian)
                        * (
                            self.phase_off[row_index, column_index]
                            - self.phase_off[row_index + 1, column_index]
                        )
                        ** 2
                    )
                    + self.pairwise_bias
                )

                current_index = self.row * column_index + row_index

                self.graph.add_edge(
                    current_index, 1 + current_index, capacity=int(capacity)
                )
                self.graph.add_edge(
                    1 + current_index, current_index, capacity=int(capacity)
                )


class _TLink(_NLink):
    def __init__(self):
        super().__init__()

    def add_t_links(self):
        # data項のedge
        for row_index in range(self.row):
            for column_index in range(self.column):
                current_index = self.row * column_index + row_index
                self.graph.add_edge(
                    "s",
                    current_index,
                    capacity=self.dataterm_weight
                    * self.dataterm[row_index, column_index],
                )
                self.graph.add_edge(
                    current_index,
                    "t",
                    capacity=self.dataterm_weight
                    * (255 - self.dataterm[row_index, column_index]),
                )

        # 前景seed
        x, y = np.where((self.detection_result > 200) | (self.backprop_result > 3))
        for index in zip(x, y):
            current_index = self.row * index[1] + index[0]
            self.graph.add_edge("s", current_index, capacity=1_000_000)

        # 背景seed
        # x, y = np.where(self. < 0.01)
        # for i in range(x.shape[0]):
        #     temp = y[i] * self.column + x[i]
        #     self.graph.add_edge(temp, "t", capacity=1_000_000)
        # self.graph.add_edge(1, "t", capacity=1_000_000)


class GraphCut(_TLink):
    def __init__(self):
        super().__init__()

    def graphcut(self):
        self.create_graph()
        # minimum_cut
        cut_value, partition = nx.minimum_cut(self.graph, "s", "t")
        s, t = partition
        s = list(s)

        # 二次元が一次元になっているので画像表示のため
        result = np.zeros((self.row, self.column))
        for i in s:
            if i != "s":
                if i == 0:
                    result[0, 0] = 1
                else:
                    column_n = i // self.row
                    row_n = i % self.row
                    result[row_n - 1, column_n - 1] = 1
        return result

    def create_graph(self):
        # 有向グラフを定義
        self.graph = nx.DiGraph()

        # row*column個のnodeを定義
        N = self.column * self.row
        self.graph.add_nodes_from(range(N))

        # 行方向、列方向のedgeを伸ばす
        self.column_n_links()
        self.row_n_links()

        # source,sinkからのedgeを伸ばす
        self.graph.add_nodes_from(["s", "t"])
        self.add_t_links()
