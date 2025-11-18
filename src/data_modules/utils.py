# src/utils.py
import torch
import numpy as np
from skimage.filters import threshold_otsu
from skimage import measure


def dice_score(true, pred):
    """Calculate Dice score"""
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = true * pred
    denom = true + pred

    if true.sum() == 0 and pred.sum() > 0:
        return 0.0
    elif true.sum() > 0 and pred.sum() == 0:
        return 0.0
    elif true.sum() == 0 and pred.sum() == 0:
        return 1.0

    return 2.0 * np.sum(inter) / np.sum(denom)


def f_beta_score(pred, gt, beta2=0.3, threshold=0.5):
    """Calculate F-beta score"""
    pred_bin = (pred >= threshold).astype(np.uint8)

    if pred.max() == 0 and gt.max() == 0:
        return 1.0

    TP = np.sum((pred_bin == 1) & (gt == 1))
    FP = np.sum((pred_bin == 1) & (gt == 0))
    FN = np.sum((pred_bin == 0) & (gt == 1))

    if TP == 0:
        return 0.0

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)

    f_beta = ((1 + beta2) * precision * recall) / (
        beta2 * precision + recall + 1e-8
    )
    return float(f_beta)


def labnorm(image):
    """Normalize LAB image"""
    new_image = torch.empty_like(image, dtype=image.dtype)

    new_image[:, 0, :, :] = image[:, 0, :, :] / 99.998337
    new_image[:, 1, :, :] = (image[:, 1, :, :] + 86.182236) / (
        86.182236 + 98.258614
    )
    new_image[:, 2, :, :] = (image[:, 2, :, :] + 107.867744) / (
        107.867744 + 94.481682
    )

    return new_image


def filter_component_by_area(saliency, area_range=[1800, 10000]):
    """Filter connected components by area"""
    bin_sal = np.copy(saliency)
    thresh = threshold_otsu(saliency)
    bin_sal[saliency > thresh] = 1
    bin_sal[saliency <= thresh] = 0
    saliency[bin_sal == 0] = 0

    sal_components_image = measure.label(bin_sal, background=0, connectivity=2)
    sal_nb_components = sal_components_image.max()

    for c_ in range(1, sal_nb_components + 1):
        area = len(sal_components_image[sal_components_image == c_])

        if area < area_range[0] or area > area_range[1]:
            saliency[sal_components_image == c_] = 0

    return saliency