import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2


def plot_3d(img):
    x = np.arange(img.shape[0])
    y = np.arange(img.shape[1])

    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, img, cmap="bwr", linewidth=0)
    fig.colorbar(surf)

    # Plot a basic wireframe.
    # ax.plot_wireframe(X, Y, img, rstride=10, cstride=10)

    ax.set_title("Surface Plot")
    fig.show()

def gaus_filter(img, kernel_size, sigma):
    pad_size = int(kernel_size - 1 / 2)
    img_t = np.pad(img, (pad_size, pad_size), 'constant')  # zero padding(これしないと正規化後、画像端付近の尤度だけ明るくなる)
    img_t = cv2.GaussianBlur(img_t, ksize=(kernel_size, kernel_size), sigmaX=sigma)  # filter gaussian(適宜パラメータ調整)
    img_t = img_t[pad_size:-pad_size, pad_size:-pad_size]  # remove padding
    return img_t

