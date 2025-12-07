import os
from os.path import join as pjoin
import sys

os.system(f"make iftSymmISF_UniformGridSampling -C $NEWIFT_DIR/demo/BrainAsymmetryAnalysis/TBME/exp01_why_good_supervoxels")
os.system(f"make iftExtractSupervoxelHAAFeats -C $NEWIFT_DIR/demo/BrainAsymmetryAnalysis/SAAD")


def pkey(img_path: str):
    return img_path.split("/")[-1].split(".nii.gz")[0]


def main():
    train_set = "exps/sets/train_set_regs_cropped.csv"
    normal_asym_map_path = "./exps/normal_asym_map.nii.gz"
    right_hem_mask = "./template/cropped/mni152_nonlinear_sym_right_hemisphere.nii.gz"

    split = 1

    test_set_path = f"exps/sets/ATLAS-304/3T/{split:02d}/test_set.csv"

    with open(test_set_path, "r") as f:
        test_set = list(map(lambda line: line.split("\n")[0], f.readlines()))
        print(test_set)

    print(f"### SPLIT: {split:02d}")
    n_svoxels_list = [100, 250, 400, 550, 700]

    out_root_dir = f"exps/01_why_good_supervoxels/uniform_grid_sampling/{split:02d}"
    if not os.path.exists(out_root_dir):
        os.makedirs(out_root_dir)

    for n_svoxels in n_svoxels_list:
        print(f"# Num SVoxels: {n_svoxels}")
        out_eval_dir = pjoin(out_root_dir, f"n_svoxels_{n_svoxels:03d}")

        for img_path in test_set:
            img_pkey = pkey(img_path)
            print(f"\t- Image: {img_pkey}")

            out_img_root_dir = pjoin(out_eval_dir, img_pkey)
            if not os.path.exists(out_img_root_dir):
                os.makedirs(out_img_root_dir)

            print(f"\t\t- Supervoxel Extraction")
            # supervoxel extraction
            out_svoxels_path = pjoin(out_img_root_dir, "svoxels.nii.gz")
            os.system(f"iftSymmISF_UniformGridSampling --image-path {img_path} --binary-mask {right_hem_mask} "
                      f"--number-of-supervoxels {n_svoxels} --output-supervoxel-image-path {out_svoxels_path} "
                      f"--alpha 0.08 --beta 3")

            print(f"\t\t- Feature Extraction")
            out_datasets_dir = pjoin(out_img_root_dir, "datasets")
            os.system(f"iftExtractSupervoxelHAAFeats --test-image {img_path} --test-symmetric-supervoxels {out_svoxels_path} "
                      f"--training-set {train_set} --output-dir {out_datasets_dir} --num-of-bins 128 "
                      f"--normal-asymmetry-map {normal_asym_map_path}")

            print(f"\t\t- Anomaly Classification")
            output_anomalous_supervoxels_img = pjoin(out_img_root_dir, "result.nii.gz")

            os.system(f"python $NEWIFT_DIR/demo/BrainAsymmetryAnalysis/SAAD/classify_asymmetries_by_ocsvm.py " \
                      f"--symmetric-supervoxels-path {out_svoxels_path} " \
                      f"--supervoxels-datasets-entry {out_datasets_dir} " \
                      f"--output-anomalous-supervoxels-img {output_anomalous_supervoxels_img} -k linear -n 0.1")
        print("")
    print("")


if __name__ == "__main__":
    main()
