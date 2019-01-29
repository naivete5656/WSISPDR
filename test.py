import matplotlib.pyplot as plt
import numpy as np
import cv2

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from scipy.ndimage.morphology import morphological_gradient



img = img_as_float(astronaut()[::2, ::2])
img = cv2.imread('./images/for_graphcut/test18_normalize/backprop/00000.tif')
img = cv2.imread('./images/sequence/cuts/test18/00000.tif')
# img = 255 - img
img = img.astype(np.float64) / 255
segments_fz = felzenszwalb(img, scale=1000, sigma=0.5, min_size=50)
segments_slic = slic(img, n_segments=1000, compactness=10, sigma=1)
segments_quick = quickshift(img, kernel_size=5, max_dist=6, ratio=0.5)
gradient = sobel(rgb2gray(img))
segments_watershed = watershed(gradient, markers=1000, compactness=0.001)

print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0, 0].imshow(mark_boundaries(img, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")
ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(mark_boundaries(img, segments_quick))
ax[1, 0].set_title('Quickshift')
ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
ax[1, 1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()