import numpy as np


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])


def normalize(x):
    return x / 255


def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b



