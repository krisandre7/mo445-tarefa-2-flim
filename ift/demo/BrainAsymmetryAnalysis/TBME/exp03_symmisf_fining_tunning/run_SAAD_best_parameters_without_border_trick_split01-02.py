import os
from os.path import join as pjoin
import shutil


def pkey(img_path: str):
    return img_path.split("/")[-1].split(".nii.gz")[0]



# The best parameters (for this version without border trick and mean+std normal asymmetry map) were found in previous
# experiments.


def main():
    train_set_dir = "./bases/CamCan428/regs/nonrigid/final/T1"
    normal_asym_map_path = "./exps/normal_asym_map.nii.gz"
    right_hem_mask = "./template/mni152_nonlinear_sym_right_hemisphere_close.nii.gz"

    for it in [1, 2]:
        test_set_path = f"sets/ATLAS-304/{it:02d}/test_set_regs.csv"

        with open(test_set_path, "r") as f:
            test_set = list(map(lambda line: line.split("\n")[0], f.readlines()))
            print(test_set)

        print(f"### SPLIT: {it:02d}")
        out_root_dir = f"exps/03_symmisf_fining_tunning/SAAD_best_parameters_without_border_trick/split_{it:02d}"
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
                  f"{img_path} {right_hem_mask} {train_set_dir} {output_anomalous_supervoxels_img} " \
                  f"--normal-asymmetry-map {normal_asym_map_path} --symmisf_alpha 0.12 --symmisf_beta 5 " \
                  f"--symmisf_otsu_threshold_factor 0.5 --symmisf_min_euclidean_distance 0.0 " \
                  f"--symmisf_num_seeds_on_symmetric_regions 100 --bins 128 --ocsvm_kernel linear " \
                  f"--ocsvm_nu 0.1 --aux_dir {out_img_root_dir}"
            os.system(cmd)
        print("")


if __name__ == "__main__":
    main()
