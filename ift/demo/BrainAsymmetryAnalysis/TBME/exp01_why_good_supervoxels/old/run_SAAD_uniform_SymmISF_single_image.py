import os
from os.path import join as pjoin
import sys
import time

os.system(f"make iftSymmISF_UniformGridSampling -C $NEWIFT_DIR/demo/BrainAsymmetryAnalysis/NewPaper/exp01_why_good_supervoxels")
os.system(f"make iftExtractSupervoxelHAAFeats -C $NEWIFT_DIR/demo/BrainAsymmetryAnalysis/SAAD")


def pkey(img_path: str):
    return img_path.split("/")[-1].split(".nii.gz")[0]


def main():
    if len(sys.argv) != 4:
        sys.exit(f"python {sys.argv[0]} <reg_image.nii.gz> <num_supervoxels> <out_root_dir>")

    img_path = sys.argv[1]
    n_svoxels = int(sys.argv[2])
    out_root_dir = sys.argv[3]
    if not os.path.exists(out_root_dir):
        os.makedirs(out_root_dir)

    train_set_dir = "./bases/CamCan428/regs/nonrigid/final/T1"
    normal_asym_map_path = "./exps/normal_asym_map.nii.gz"
    right_hem_mask = "./template/mni152_nonlinear_sym_right_hemisphere_close.nii.gz"

    start = time.time()

    print(f"\t\t- Supervoxel Extraction")
    # supervoxel extraction
    out_svoxels_path = pjoin(out_root_dir, "svoxels.nii.gz")
    os.system(f"iftSymmISF_UniformGridSampling --image-path {img_path} --binary-mask {right_hem_mask} "
              f"--number-of-supervoxels {n_svoxels} --output-supervoxel-image-path {out_svoxels_path} "
              f"--alpha 0.08 --beta 3")

    print(f"\t\t- Feature Extraction")
    out_datasets_dir = pjoin(out_root_dir, "datasets")
    os.system(f"iftExtractSupervoxelHAAFeats --test-image {img_path} --test-symmetric-supervoxels {out_svoxels_path} "
              f"--training-set {train_set_dir} --output-dir {out_datasets_dir} --num-of-bins 128 "
              f"--normal-asymmetry-map {normal_asym_map_path}")

    print(f"\t\t- Anomaly Classification")
    output_anomalous_supervoxels_img = pjoin(out_root_dir, "result.nii.gz")

    os.system(f"python $NEWIFT_DIR/demo/BrainAsymmetryAnalysis/SAAD/classify_asymmetries_by_ocsvm.py {out_svoxels_path} "
              f"{out_datasets_dir} {output_anomalous_supervoxels_img} -k linear -n 0.1")

    end = time.time()
    print(f"\n==> Processing Time: {end - start} secs")


if __name__ == "__main__":
    main()
