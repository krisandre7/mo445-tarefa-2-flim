import os
import random
import sys

import numpy as np

import pyift.pyift as ift


def select_samples_by_svoxels_and_gt(svoxels, gt, sampling_voxel_perc_per_svoxel=0.05):
    svoxels_ = np.array(svoxels)
    n_svoxels = svoxels.max()

    # gets all voxels on GT
    gt_voxels = np.where(gt != 0)[0]
    svoxels_[gt_voxels] = 0  # remove (set to zero) the gt voxels on svoxels map to avoid sampling duplication ahead

    selected_samples_idxs = list(gt_voxels)

    for label in range(1, n_svoxels + 1):
        label_idxs = list(np.where(svoxels_ == label)[0])
        random.shuffle(label_idxs)

        n_selected_samples = int(sampling_voxel_perc_per_svoxel * len(label_idxs))
        selected_samples_idxs.extend(label_idxs[:n_selected_samples])

    print(f"len(selected_samples_idxs) = {len(selected_samples_idxs)}")
    print(f"len(set(selected_samples_idxs)) = {len(set(selected_samples_idxs))}")

    return np.array(selected_samples_idxs)


def main():
    if len(sys.argv) != 6:
        sys.exit(f"python {sys.argv[0]} <symm_voxel_hist_feats.zip> <symm_voxel_coord_and_hist_feats.zip> "
                 "<sampling_voxel_perc_per_svoxel> <out_symm_voxel_hist_feats_selected.zip> "
                 "<out_symm_voxel_coord_and_hist_feats_selected>")

    symm_voxel_hist_feats_path = sys.argv[1]
    symm_voxel_coord_and_hist_feats_path = sys.argv[2]
    sampling_voxel_perc_per_svoxel = float(sys.argv[3])
    out_symm_voxel_hist_feats_selected = sys.argv[4]
    out_symm_voxel_coord_and_hist_feats_selected = sys.argv[5]

    parent_dir = os.path.dirname(out_symm_voxel_hist_feats_selected)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    print("------------------------------------------")
    print(f"- Symmetric Voxel Hist Feats: {symm_voxel_hist_feats_path}")
    print(f"- Symmetric Voxel Coord. and Hist Feats: {symm_voxel_coord_and_hist_feats_path}")
    print(f"- Sampling Voxel Percentage per Supervoxel: {sampling_voxel_perc_per_svoxel}")
    print(f"- Output Symmetric Voxel Hist Feats: {out_symm_voxel_hist_feats_selected}")
    print(f"- Output Symmetric Voxel Coord. Hist Feats: {out_symm_voxel_coord_and_hist_feats_selected}")
    print("------------------------------------------\n")

    print('- Reading Datasets')
    Zhist = ift.ReadDataSet(symm_voxel_hist_feats_path)
    Xhist = Zhist.GetData()
    print(f"(n_samples, n_feats) = {Xhist.shape}")
    svoxels = Zhist.GetGroups()
    gts = Zhist.GetTrueLabels()

    Zcoord_hist = ift.ReadDataSet(symm_voxel_coord_and_hist_feats_path)
    Xcoord_hist = Zcoord_hist.GetData()

    if np.all(svoxels == Zcoord_hist.GetGroups()) is False:
        sys.exit("Supervoxel array is different")
    if np.all(gts == Zcoord_hist.GetTrueLabels()) is False:
        sys.exit("GT lesions array is different")

    selected_samples_idxs = select_samples_by_svoxels_and_gt(svoxels, gts, sampling_voxel_perc_per_svoxel)

    svoxels = svoxels[selected_samples_idxs]
    gts = gts[selected_samples_idxs]

    Xhist = Xhist[selected_samples_idxs]
    Zhist = ift.CreateDataSetFromNumPy(Xhist, gts)
    Zhist.SetGroups(svoxels)
    ift.WriteDataSet(Zhist, out_symm_voxel_hist_feats_selected)
    print(f"sampled (n_samples, n_feats) = {Xhist.shape}")

    Xcoord_hist = Xcoord_hist[selected_samples_idxs]
    Zcoord_hist = ift.CreateDataSetFromNumPy(Xcoord_hist, gts)
    Zcoord_hist.SetGroups(svoxels)
    ift.WriteDataSet(Zcoord_hist, out_symm_voxel_coord_and_hist_feats_selected)
    print(f"sampled (n_samples, n_feats) = {Xcoord_hist.shape}")


if __name__ == "__main__":
    main()
