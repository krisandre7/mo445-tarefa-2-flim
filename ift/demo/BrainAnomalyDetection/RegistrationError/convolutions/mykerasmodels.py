import sys

import numpy as np
from keras.models import Sequential
from keras.layers import ZeroPadding3D, Conv3D


def build_kernel_bank(n_kernels=32, kernel_side_size=5):
    shape = (kernel_side_size, kernel_side_size,
             kernel_side_size, 1)
    kernel_bank = np.zeros((*shape, n_kernels))

    for i in range(n_kernels):
        kernel = np.random.uniform(size=shape)
        kernel = kernel - kernel.mean()  # zero mean
        kernel = kernel / np.linalg.norm(kernel)  # unit norm
        kernel_bank[:, :, :, :, i] = kernel

    return kernel_bank


def build_simple_conv3D_model(input_shape, n_kernels=32, kernel_side_size=5,
                              activation="linear"):
    if kernel_side_size % 2 == 0:
        sys.exit(
            f"Only allowed odd kernel side size. Found {kernel_side_size}")

    input_shape_ext = (*input_shape, 1)

    kernel_bank = build_kernel_bank(n_kernels, kernel_side_size)

    model = Sequential()
    model.add(ZeroPadding3D(kernel_side_size // 2, input_shape=input_shape_ext))
    model.add(Conv3D(n_kernels, kernel_side_size, use_bias=False,
                     activation=activation))
    model.layers[-1].set_weights([kernel_bank])

    return model


def predict(img_data, model):
    # (zsize, ysize, xsize, 1)
    img_data = np.reshape(img_data, (*img_data.shape, 1))

    # (zsize, ysize, xsize, n_channels)
    input_shape = model.layers[0].input_shape[1:]

    if img_data.shape != input_shape:
        sys.exit("Image with Different Shape of the Model's Input Layer")

    # (1, zsize, ysize, xsize, n_channels) ==> 1 == n_samples to predict
    img_data = np.reshape(img_data, (1, *img_data.shape))
    filt_data = model.predict(img_data)
    filt_data = np.reshape(filt_data, filt_data.shape[1:])

    return filt_data
