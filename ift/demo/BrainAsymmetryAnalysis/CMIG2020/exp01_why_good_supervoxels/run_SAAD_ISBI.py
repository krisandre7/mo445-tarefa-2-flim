import os
from os.path import join as pjoin
import shutil
import sys


def pkey(img_path: str):
    return img_path.split("/")[-1].split(".nii.gz")[0]


def main():
    n_iters = 5
    train_set = "exps/sets/train_set_regs_cropped.csv"
    normal_asym_map_path = "./exps/normal_asym_map.nii.gz"
    right_hem_mask = "./template/cropped/mni152_nonlinear_sym_right_hemisphere.nii.gz"

    for it in range(1, n_iters + 1):
        test_set_path = f"exps/sets/ATLAS-304/3T/{it:02d}/test_set.csv"

        with open(test_set_path, "r") as f:
            test_set = sorted(list(map(lambda line: line.split("\n")[0], f.readlines())))
            print(test_set)

        print(f"### SPLIT: {it:02d}")
        out_root_dir = f"exps/01_why_good_supervoxels/SAAD_ISBI/{it:02d}"
        if not os.path.exists(out_root_dir):
            os.makedirs(out_root_dir)

        for img_path in test_set:
            img_pkey = pkey(img_path)
            print(f"\t- Image: {img_pkey}")

            out_img_root_dir = pjoin(out_root_dir, img_pkey)
            if not os.path.exists(out_img_root_dir):
                os.makedirs(out_img_root_dir)

            output_anomalous_supervoxels_img = pjoin(out_img_root_dir, "result.nii.gz")

            cmd = f"python $NEWIFT_DIR/demo/BrainAsymmetryAnalysis/SAAD/run_SAAD.py " \
                  f"--image-path {img_path} --right-hemisphere-mask-path {right_hem_mask} " \
                  f"--train-image-set-entry {train_set} --out-classification-mask {output_anomalous_supervoxels_img} " \
                  f"--normal-asymmetry-map {normal_asym_map_path} --symmisf_alpha 0.08 --symmisf_beta 3 " \
                  f"--symmisf_otsu_threshold_factor 2.0 --symmisf_min_euclidean_distance 0.0 " \
                  f"--symmisf_num_seeds_on_symmetric_regions 100 --bins 128 --ocsvm_kernel linear " \
                  f"--ocsvm_nu 0.1 --aux_dir {out_img_root_dir}"
            os.system(cmd)
        print("")


if __name__ == "__main__":
    main()
