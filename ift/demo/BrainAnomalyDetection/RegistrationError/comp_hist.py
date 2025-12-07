import timeit

import numpy as np

import pyift.pyift as ift


data = ift.ReadImageByExt(
    "/Users/hisamuka/workspace/exps/2020_Martins_ISBI_AnomalyDetectionTemplateDifference/bases/ATLAS-304/3T/regs/nonrigid/final/000209_000001.nii.gz").AsNumPy()
mask = ift.ReadImageByExt(
    "/Users/hisamuka/workspace/exps/2020_Martins_ISBI_AnomalyDetectionTemplateDifference/template/mni152_nonlinear_sym_brain_close.nii.gz").AsNumPy()


n_labels = mask.max()

t1 = timeit.default_timer()

for label in range(1, n_labels + 1):
    # print(f"label: {label}")
    hist = np.histogram(data[mask == label], bins=128)

print(f"\nDone in {timeit.default_timer() - t1} secs")

t1 = timeit.default_timer()

hist = [np.histogram(data[mask == label], bins=128)
        for label in range(1, n_labels + 1)]

print(f"\nDone in {timeit.default_timer() - t1} secs")


