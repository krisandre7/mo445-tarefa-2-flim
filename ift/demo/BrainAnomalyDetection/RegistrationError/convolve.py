import tensorflow as tf
from keras.models import Sequential
from keras.layers import ZeroPadding3D, Conv3D
import numpy as np
import pyift.pyift as ift

import timeit


def create_kernel_bank(n_kernels=32, kernel_side_size=5):
    shape = (kernel_side_size, kernel_side_size,
             kernel_side_size, 1)
    kernel_bank = np.zeros((*shape, n_kernels))

    for i in range(n_kernels):
        kernel = np.random.uniform(size=shape)
        kernel = kernel - kernel.mean()
        kernel = kernel / np.linalg.norm(kernel)
        kernel_bank[:, :, :, :, i] = kernel

    return kernel_bank


img = ift.ReadImageByExt(
    "/Users/hisamuka/workspace/exps/2020_Martins_ISBI_AnomalyDetectionTemplateDifference/bases/ATLAS-304/3T/regs/nonrigid/final/000096_000001.nii.gz")
label_img = ift.ReadImageByExt(
    "/Users/hisamuka/workspace/exps/2020_Martins_ISBI_AnomalyDetectionTemplateDifference/template/mni152_nonlinear_sym_brain_close.nii.gz")
template = ift.ReadImageByExt(
    "/Users/hisamuka/workspace/exps/2020_Martins_ISBI_AnomalyDetectionTemplateDifference/template/mni152_nonlinear_sym_hist.nii.gz")

bb = ift.MinBoundingBox(label_img, None)
img_roi_data = ift.ExtractROI(img, bb).AsNumPy()
img_roi_data = np.reshape(img_roi_data, (*img_roi_data.shape, 1))
template_roi_data = ift.ExtractROI(template, bb).AsNumPy()
template_roi_data = np.reshape(
    template_roi_data, (*template_roi_data.shape, 1))

kernel_side_size = 5
n_kernels = 100


kernel_bank = create_kernel_bank(n_kernels, kernel_side_size)


print(img_roi_data.shape)
print(kernel_bank.shape)


model = Sequential()
model.add(ZeroPadding3D(kernel_side_size // 2, input_shape=img_roi_data.shape))
model.add(Conv3D(n_kernels, kernel_side_size,
                 use_bias=False, activation="relu"))

print(len(model.layers))
print(model.layers[-1].get_weights()[0].shape)
model.layers[-1].set_weights([kernel_bank])
print(model.layers[-1].get_weights()[0].shape)
# print(len(model.layers[-1].get_weights()))

filt = model.predict(np.reshape(img_roi_data, (1, *img_roi_data.shape)))
filt = np.reshape(filt, tuple(filt.shape[1:]))

print("Saving")
for i in range(n_kernels):
    filt_roi_data = filt[:, :, :, i].astype(np.int32)
    filt_roi_img = ift.CreateImageFromNumPy(filt_roi_data, True)
    filt_img = ift.CreateImageFromImage(img)
    ift.InsertROI(filt_roi_img, filt_img, bb.begin)
    filt_img.Write(f"tmp/kernel_{i}/filt_img.nii.gz")
print("Done")
