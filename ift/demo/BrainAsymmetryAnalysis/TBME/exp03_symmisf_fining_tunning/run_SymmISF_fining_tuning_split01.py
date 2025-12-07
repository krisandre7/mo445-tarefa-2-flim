import os
from os.path import join as pjoin
import shutil


def main():
    normal_asym_map_path = "./exps/normal_asym_map_std.nii.gz"
    right_hem_mask = "./template/mni152_nonlinear_sym_right_hemisphere_close.nii.gz"
    train_set_dir = "./bases/CamCan428/regs/nonrigid/final/T1"

    alpha_grid = [0.06, 0.08, 0.1]
    beta_grid = [3, 5, 7]
    thres_factor_grid = [0.5, 1.0, 1.5, 2.0]

    it = 1  # split
    eval_img_set_path = f"sets/ATLAS-304/{it:02d}/eval_set_regs.csv"

    with open(eval_img_set_path, "r") as f:
        eval_img_set = list(map(lambda line: line.split("\n")[0], f.readlines()))
        print(eval_img_set)

    print(f"### SPLIT: {it:02d}")
    out_root_dir = f"exps/03_symmisf_fining_tunning/split_{it:02d}"
    if not os.path.exists(out_root_dir):
        os.makedirs(out_root_dir)

    for alpha in alpha_grid:
        for beta in beta_grid:
            for thres_factor in thres_factor_grid:
                out_grid_dir = pjoin(out_root_dir, f"alpha-{alpha}_beta-{beta}_thres-{thres_factor}")

                for img_path in eval_img_set:
                    img_key = img_path.split("/")[-1].split(".nii.gz")[0]
                    out_img_root_dir = pjoin(out_grid_dir, img_key)
                    output_anomalous_supervoxels_img = pjoin(out_img_root_dir, f"result.nii.gz")

                    cmd = f"python $NEWIFT_DIR/demo/BrainAsymmetryAnalysis/SAAD/run_SAAD.py " \
                          f"{img_path} {right_hem_mask} {train_set_dir} {output_anomalous_supervoxels_img} " \
                          f"--normal-asymmetry-map {normal_asym_map_path} --symmisf_alpha {alpha} --symmisf_beta {beta} " \
                          f"--symmisf_otsu_threshold_factor {thres_factor} --symmisf_min_euclidean_distance 5.0 " \
                          f"--symmisf_num_seeds_on_symmetric_regions 50 --bins 128 --ocsvm_kernel linear " \
                          f"--ocsvm_nu 0.1 --aux_dir {out_img_root_dir}"
                    os.system(cmd)
                    shutil.rmtree(pjoin(out_img_root_dir, f"datasets"))
            print("")
            print("")
        print("")


if __name__ == "__main__":
    main()
