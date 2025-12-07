import glob
import os
import sys

import numpy as np
import pyift.pyift as ift


def validateInputs(result_dir, gt_dir, out_csv_path):
    if not os.path.exists(result_dir):
        sys.exit(f"Result Dir: {result_dir} does not exists")
    if not os.path.exists(gt_dir):
        sys.exit(f"GT Dir: {result_dir} does not exists")
    if not out_csv_path.lower().endswith(".csv"):
        sys.exit(f"Invalid CSV file: {out_csv_path}")

    parent_dir = os.path.dirname(out_csv_path)
    if not parent_dir or not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


def main():
    if len(sys.argv) != 4:
        sys.exit(f"python {sys.argv[0]} <result_dir> <gt_dir> <out_csv_path>\n"
                 "E.g:\n"
                 "result_dir: + thesis_experiments/results/affine_affine/01\n"
                 "                + 000003_000001.nii.gz\n"
                 "                + 000005_000001.nii.gz\n"
                 "                ...\n"
                 "gt_dir: + bases/ATLAS-304/native/labels/primary_stroke\n"
                 "            + 000001_000001.nii.gz\n"
                 "            + 000002_000001.nii.gz\n"
                 "            ...\n")

    result_dir = sys.argv[1]
    gt_dir = sys.argv[2]
    out_csv_path = sys.argv[3]
    validateInputs(result_dir, gt_dir, out_csv_path)

    out_csv = open(out_csv_path, "w")
    out_csv.write("detection_image_path,object,dice\n")

    dice_list = []

    for detection_img_path in glob.glob(f"{result_dir}/*"):
        image_filename = os.path.basename(detection_img_path)
        gt_path = os.path.join(gt_dir, image_filename)
        
        print(f"Detection Image: {detection_img_path}")
        print(f"GT: {gt_path}")

        detection_data = ift.ReadImageByExt(detection_img_path).AsNumPy()
        detection_data_bin = detection_data != 0
        detection_volume = np.sum(detection_data_bin)
        # ift.CreateImageFromNumPy(detection_data_bin.astype(
        #     "int32"), True).Write(f"tmp/{image_key}_detection_data_bin.nii.gz")

        gt_data = ift.ReadImageByExt(gt_path).AsNumPy()
        gt_data_bin = gt_data != 0
        gt_volume = np.sum(gt_data_bin)
        # ift.CreateImageFromNumPy(gt_data_bin.astype(
        #     "int32"), True).Write(f"tmp/{image_key}_gt_data_bin.nii.gz")

        intersec_data_bin = detection_data_bin * gt_data_bin
        intersect_volume = np.sum(intersec_data_bin)
        # ift.CreateImageFromNumPy(intersec_data_bin.astype(
        #     "int32"), True).Write(f"tmp/{image_key}_intersec_data_bin.nii.gz")

        dice = (2 * intersect_volume) / (gt_volume + detection_volume)
        dice_list.append(dice)

        out_csv.write(f"{detection_img_path},{dice}\n")

        print(f"\t\tDice: {dice}")
        print(f"\t\tdice = (2 * {intersect_volume}) / ({gt_volume} + {detection_volume}))")
        print("")

    dice_arr = np.array(dice_list)
    dice_mean = np.mean(dice_arr)
    dice_std = np.std(dice_arr)

    print(f"----------------------")
    print(f"# Mean DICE: {dice_mean} +- {dice_std}")

    out_csv.close()

if __name__ == "__main__":
    main()
