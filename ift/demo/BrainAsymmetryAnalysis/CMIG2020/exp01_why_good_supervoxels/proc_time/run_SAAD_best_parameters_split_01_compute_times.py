import os
from os.path import join as pjoin
import shutil
import time

import numpy as np

def pkey(img_path: str):
    return img_path.split("/")[-1].split(".nii.gz")[0]



# The best parameters (for this version without border trick and mean+std normal asymmetry map) were found in previous
# experiments.


def main():
    train_set = "exps/sets/train_set_regs_cropped.csv"
    normal_asym_map_path = "./exps/normal_asym_map_std.nii.gz"
    right_hem_mask = "./template/cropped/mni152_nonlinear_sym_right_hemisphere.nii.gz"

    test_set_path = f"exps/sets/ATLAS-304/3T/01/test_set.csv"

    with open(test_set_path, "r") as f:
        test_set = list(map(lambda line: line.split("\n")[0], f.readlines()))
        print(test_set)

    print(f"### SPLIT: 01")
    out_root_dir = f"tmp/SAAD_best_parameters"
    if not os.path.exists(out_root_dir):
        os.makedirs(out_root_dir)
    
    proc_times = []


    for img_path in test_set[:15]:
        img_pkey = pkey(img_path)
        print(f"\t- Image: {img_pkey}")

        out_img_root_dir = pjoin(out_root_dir, img_pkey)
        if not os.path.exists(out_img_root_dir):
            os.makedirs(out_img_root_dir)

        output_anomalous_supervoxels_img = pjoin(out_img_root_dir, "result.nii.gz")

        start = time.time()

        cmd = f"python $NEWIFT_DIR/demo/BrainAsymmetryAnalysis/SAAD/run_SAAD.py " \
                f"--image-path {img_path} --right-hemisphere-mask-path {right_hem_mask} " \
                f"--train-image-set-entry {train_set} --out-classification-mask {output_anomalous_supervoxels_img} " \
                f"--normal-asymmetry-map {normal_asym_map_path} --symmisf_alpha 0.06 --symmisf_beta 5 " \
                f"--symmisf_otsu_threshold_factor 0.5 --symmisf_min_euclidean_distance 5.0 " \
                f"--symmisf_num_seeds_on_symmetric_regions 100 --bins 128 --ocsvm_kernel linear " \
                f"--ocsvm_nu 0.1 --aux_dir {out_img_root_dir}"
        os.system(cmd)

        end = time.time()
        proc_times.append(end - start)
    print("")

    proc_times = np.array(proc_times)
    print(f"\n===> Mean Processing Time: {np.mean(proc_times)} +- {np.std(proc_times)} secs\n\n")

    shutil.rmtree(out_root_dir)


if __name__ == "__main__":
    main()
