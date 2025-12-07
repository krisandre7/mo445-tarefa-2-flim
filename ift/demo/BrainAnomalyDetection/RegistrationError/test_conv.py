import numpy as np
import pyift.pyift as ift

import convolve

import timeit


img = ift.ReadImageByExt(
    "/Users/hisamuka/workspace/exps/2020_Martins_ISBI_AnomalyDetectionTemplateDifference/bases/ATLAS-304/3T/regs/nonrigid/final/000096_000001.nii.gz")
label_img = ift.ReadImageByExt(
    "/Users/hisamuka/workspace/exps/2020_Martins_ISBI_AnomalyDetectionTemplateDifference/template/mni152_nonlinear_sym_brain_close.nii.gz")
template = ift.ReadImageByExt(
    "/Users/hisamuka/workspace/exps/2020_Martins_ISBI_AnomalyDetectionTemplateDifference/template/mni152_nonlinear_sym_hist.nii.gz")

bb = ift.MinBoundingBox(label_img, None)
img_roi_data = ift.ExtractROI(img, bb).AsNumPy()
template_roi_data = ift.ExtractROI(template, bb).AsNumPy()

total_t1 = timeit.default_timer()

t1 = timeit.default_timer()
kernel_shape_size = 5
kernel_bank = convolve.create_kernel_bank(n_kernels=128,
    shape=(kernel_shape_size, kernel_shape_size, kernel_shape_size))
print(f"Kernel bank: {timeit.default_timer() - t1} secs")
    
t1 = timeit.default_timer()
img_adj_mat = convolve.build_adj_image_matrix(img_roi_data, kernel_shape_size)
print(f"Build Adj Image Matrix: {timeit.default_timer() - t1} secs")

t1 = timeit.default_timer()
filt_img = np.dot(kernel_bank, img_adj_mat)
print(f"filt_img.shape = {filt_img.shape}")
print(f"Dot: {timeit.default_timer() - t1} secs")

print(f"\n==> Total: {timeit.default_timer() - total_t1} secs")
