import os
from os.path import join as pjoin
import shutil

os.system(f"make iftSymmISF -C $NEWIFT_DIR/demo/BrainAsymmetryAnalysis/SAAD")
os.system(f"make iftExtractSupervoxelHAAFeats -C $NEWIFT_DIR/demo/BrainAsymmetryAnalysis/SAAD")


def pkey(img_path: str):
    return img_path.split("/")[-1].split(".nii.gz")[0]


def main():
    n_iters = 5
    train_set_dir = "./bases/CamCan428/regs/nonrigid/final/T1"
    normal_asym_map_path = "./exps/normal_asym_map.nii.gz"
    right_hem_mask = "./template/mni152_nonlinear_sym_right_hemisphere_close.nii.gz"

    for it in range(4, 6):
        test_set_path = f"sets/ATLAS-304/{it:02d}/test_set_regs.csv"

        with open(test_set_path, "r") as f:
            test_set = list(map(lambda line: line.split("\n")[0], f.readlines()))
            print(test_set)

        print(f"### SPLIT: {it:02d}")
        out_root_dir = f"exps/01_why_good_supervoxels/split_{it:02d}/SymmISF_best_parameters"
        if not os.path.exists(out_root_dir):
            os.makedirs(out_root_dir)

        for img_path in test_set:
            img_pkey = pkey(img_path)
            print(f"\t- Image: {img_pkey}")

            out_img_root_dir = pjoin(out_root_dir, img_pkey)
            if not os.path.exists(out_img_root_dir):
                os.makedirs(out_img_root_dir)

            print(f"\t\t- Supervoxel Extraction")
            # supervoxel extraction
            out_svoxels_path = pjoin(out_img_root_dir, "svoxels.nii.gz")
            os.system(f"iftSymmISF --image-path {img_path} --binary-mask {right_hem_mask} "
                      f"--output-supervoxel-image-path {out_svoxels_path} --alpha 1.4 --beta 5 "
                      f"--threshold-otsu-factor 0.5 --normal-asymmetry-map {normal_asym_map_path}")

            print(f"\t\t- Feature Extraction")
            out_datasets_dir = pjoin(out_img_root_dir, "datasets")
            os.system(f"iftExtractSupervoxelHAAFeats --test-image {img_path} --test-symmetric-supervoxels {out_svoxels_path} "
                      f"--training-set {train_set_dir} --output-dir {out_datasets_dir} --num-of-bins 128 "
                      f"--normal-asymmetry-map {normal_asym_map_path}")

            print(f"\t\t- Anomaly Classification")
            output_anomalous_supervoxels_img = pjoin(out_img_root_dir, "result.nii.gz")

            os.system(f"python $NEWIFT_DIR/demo/BrainAsymmetryAnalysis/SAAD/classify_asymmetries_by_ocsvm.py {out_svoxels_path} "
                      f"{out_datasets_dir} {output_anomalous_supervoxels_img} -k linear -n 0.1")

            # shutil.rmtree(out_datasets_dir)
        print("")


if __name__ == "__main__":
    main()
