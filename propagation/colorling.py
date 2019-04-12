import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import center_of_mass


def prm_visualize(prms):
    """Prediction visualization.
    """

    # helper functions
    def rgb2hsv(r, g, b):
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g - b) / df) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / df) + 120) % 360
        elif mx == b:
            h = (60 * ((r - g) / df) + 240) % 360
        s = 0 if mx == 0 else df / mx
        v = mx
        return h / 360.0, s, v

    def color_palette(N):
        cmap = np.zeros((N, 3))
        for i in range(0, N):
            uid = i
            r, g, b = 0, 0, 0
            for j in range(0, 8):
                r = np.bitwise_or(r, (((uid & (1 << 0)) != 0) << 7 - j))
                g = np.bitwise_or(g, (((uid & (1 << 1)) != 0) << 7 - j))
                b = np.bitwise_or(b, (((uid & (1 << 2)) != 0) << 7 - j))
                uid = (uid >> 3)
            cmap[i, 0] = min(r + 86, 255)
            cmap[i, 1] = min(g + 86, 255)
            cmap[i, 2] = b
        cmap = cmap.astype(np.float32) / 255
        return cmap
    if prms.shape[0] > 0:
        palette = color_palette(prms.shape[0])
        height, width = prms[0].shape
        instance_mask = np.zeros((height, width, 3), dtype=np.float32)
        peak_response_map = np.zeros((height, width, 3), dtype=np.float32)
        for idx, pred in enumerate(prms):
            prm = pred

            # peak response map
            peak_response = (prm - prm.min()) / (prm.max() - prm.min())
            mask = peak_response > 0.01
            h, s, _ = rgb2hsv(palette[idx][0], palette[idx][1], palette[idx][2])
            peak_response_map[mask, 0] = h
            peak_response_map[mask, 1] = s
            peak_response_map[mask, 2] = np.power(peak_response[mask], 0.5)

        peak_response_map = hsv_to_rgb(peak_response_map)
    return peak_response_map