#!/usr/bin/python2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import slic
import scipy
import scipy.misc

im = np.array(Image.open("grass.jpg"))
region_labels = slic.slic_n(im, 100, 10)
contours = slic.contours(im, region_labels, 10)

scipy.misc.imsave('hue.ppm', contours)

plt.imshow(contours[:, :, :-1].copy())
plt.show()
