import os
from os.path import join as pjoin
import shutil


os.system(f"make iftSymmISF_ThresFactor -C $NEWIFT_DIR/demo/BrainAsymmetryAnalysis/NewPaper/exp02_symmisf_fining_tunning")
os.system(f"make iftExtractSupervoxelHAAFeats -C $NEWIFT_DIR/demo/BrainAsymmetryAnalysis/SAAD")


def main():
    normal_asym_map_path = "./exps/normal_asym_map.nii.gz"
    right_hem_mask = "./template/mni152_nonlinear_sym_right_hemisphere_close.nii.gz"
    train_set_dir = "./bases/CamCan428/regs/nonrigid/final/T1"

    alpha_grid = [1.0, 1.2]
    beta_grid = [1, 3, 5, 7, 9]
    thres_factor_grid = [0.5, 1.0, 1.5, 2.0, 2.5]

    it = 1  # split
    eval_img_set_path = f"sets/ATLAS-304/{it:02d}/eval_set_regs.csv"

    with open(eval_img_set_path, "r") as f:
        eval_img_set = list(map(lambda line: line.split("\n")[0], f.readlines()))
        print(eval_img_set)

    print(f"### SPLIT: {it:02d}")
    out_root_dir = f"exps/02_symmisf_fining_tunning/split_{it:02d}"
    if not os.path.exists(out_root_dir):
        os.makedirs(out_root_dir)

    for alpha in alpha_grid:
        for beta in beta_grid:
            for thres_factor in thres_factor_grid:
                out_grid_dir = pjoin(out_root_dir, f"alpha-{alpha}_beta-{beta}_thres-{thres_factor}")

                for img_path in eval_img_set:
                    img_key = img_path.split("/")[-1].split(".nii.gz")[0]

                    out_svoxels_path = pjoin(out_grid_dir, img_path.split("/")[-1])
                    print(f"\t- Image: {img_path}")
                    print(f"\t- Out SVoxels: {out_svoxels_path}")

                    # os.system(f"iftSymmISF_ThresFactor --image-path {img_path} --binary-mask {right_hem_mask} "
                    #           f"--output-supervoxel-image-path {out_svoxels_path} --alpha {alpha} --beta {beta} "
                    #           f"--threshold-otsu-factor {thres_factor} --normal-asymmetry-map {normal_asym_map_path}")

                    out_datasets_dir = pjoin(out_grid_dir, f"{img_key}_datasets")
                    os.system(f"iftExtractSupervoxelHAAFeats --test-image {img_path} --test-symmetric-supervoxels {out_svoxels_path} "
                              f"--training-set {train_set_dir} --output-dir {out_datasets_dir} --num-of-bins 128 "
                              f"--normal-asymmetry-map {normal_asym_map_path}")

                    print(f"\t\t- Anomaly Classification")
                    output_anomalous_supervoxels_img = pjoin(out_grid_dir, f"{img_key}_result.nii.gz")

                    os.system(f"python $NEWIFT_DIR/demo/BrainAsymmetryAnalysis/SAAD/classify_asymmetries_by_ocsvm.py {out_svoxels_path} "
                              f"{out_datasets_dir} {output_anomalous_supervoxels_img} -k linear -n 0.1")

                    shutil.rmtree(out_datasets_dir)
            print("")
            print("")
        print("")


if __name__ == "__main__":
    main()
