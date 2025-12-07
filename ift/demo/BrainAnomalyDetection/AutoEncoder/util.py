import csv
import math
import os
import pdb
import sys

import numpy as np

import pyift.pyift as ift


def read_image_list(img_list):
    img0 = ift.ReadImageByExt(img_list[0]).AsNumPy()
    
    shape = (len(img_list), *img0.shape)
    X = np.zeros(shape)

    for i, img_path in enumerate(img_list):
        print(img_path)
        img = ift.ReadImageByExt(img_path).AsNumPy()
        X[i] = img

    new_shape = tuple(list(X.shape) + [1])
    X = np.reshape(X, new_shape)

    return X


def normalization_value(img):
    # Given that the image formats are only 8, 12, 16, 32, and 64 bits, we must impose this constraint here.
    n_bits = math.floor(math.log2(img.max()))

    if n_bits > 1 and n_bits < 8:
        n_bits = 8
    elif n_bits > 8 and n_bits < 12:
        n_bits = 12
    elif n_bits > 12 and n_bits < 16:
        n_bits = 16
    elif n_bits > 16 and n_bits < 32:
        n_bits = 32
    elif n_bits > 32 and n_bits < 64:
        n_bits = 64
    elif n_bits > 64:
        sys.exit("Error: Number of Bits %d not supported. Try <= 64" % n_bits)

    return math.pow(2, n_bits) - 1


def normalize_image_set(X):
    # convert the int images to float and normalize them to [0, 1]
    norm_val = normalization_value(X[0])

    return X.astype('float32') / norm_val


def find_fit_size(size, reduction_factor):
    reducted_size = size / reduction_factor
    increment_to_fit = reduction_factor * \
        (math.ceil(reducted_size) - reducted_size)
    fit_size = int(size + increment_to_fit)

    return fit_size


def adjust_image_domain(X, max_pooling_sizes):
    _, zsize, ysize, xsize, _ = X.shape
    reduction_factor = np.prod(max_pooling_sizes)

    fit_zsize = find_fit_size(zsize, reduction_factor)
    fit_ysize = find_fit_size(ysize, reduction_factor)
    fit_xsize = find_fit_size(xsize, reduction_factor)

    X_pad = np.pad(X, ((0, 0), (0, fit_zsize - zsize), (0, fit_ysize - ysize),
                   (0, fit_xsize - xsize), (0, 0)), mode="constant")

    return X_pad


