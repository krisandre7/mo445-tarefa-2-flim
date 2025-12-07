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
                 "result_dir: + exps/01_why_good_supervoxels/split_01/uniform_grid_sampling/n_svoxels_100\n"
                 "                + 000003_000001\n"
                 "                    - result.nii.gz\n"
                 "                ...\n"
                 "gt_dir: + bases/ATLAS-304/3T/regs/nonrigid/labels/primary_stroke\n"
                 "            + 000001_000001.nii.gz\n"
                 "            + 000002_000001.nii.gz\n"
                 "            ...\n")

    result_dir = sys.argv[1]
    gt_dir = sys.argv[2]
    out_csv_path = sys.argv[3]
    validateInputs(result_dir, gt_dir, out_csv_path)

    image_key_list = sorted(os.listdir(result_dir))

    out_csv = open(out_csv_path, "w")
    out_csv.write("detection_image_path,object,dice\n")

    dice_list = []

    for image_key in image_key_list:
        detection_img_path = os.path.join(result_dir, image_key, "result.nii.gz")
        gt_path = os.path.join(gt_dir, image_key + ".nii.gz")
        # print(f"Detection Image: {detection_img_path}")
        # print(f"GT: {gt_path}")

        detection_data = ift.ReadImageByExt(detection_img_path).AsNumPy()
        gt_data = ift.ReadImageByExt(gt_path).AsNumPy()

        n_objs = gt_data.max()

        for obj in range(1, n_objs + 1):
            gt_obj_data_bool = np.array(gt_data == obj)
            intersec_data = detection_data * gt_obj_data_bool
            intersec_data_bin = intersec_data != 0
            intersection_svoxels_arr = np.unique(intersec_data[intersec_data != 0])

            symm_detection_svoxels_data = detection_data * np.isin(detection_data, intersection_svoxels_arr)
            symm_detection_svoxels_data_bool = symm_detection_svoxels_data != 0

            gt_obj_volume = np.sum(gt_obj_data_bool)
            symm_detection_svoxels_volume = np.sum(symm_detection_svoxels_data_bool)
            intersect_volume = np.sum(intersec_data_bin)

            # we divided the symm_detection_svoxels_volume since it provides the entire supervoxels intersected
            # by the GT's object in both hemispheres
            dice = (2 * intersect_volume) / (gt_obj_volume + (symm_detection_svoxels_volume / 2))
            dice_list.append(dice)

            out_csv.write(f"{detection_img_path},{obj},{dice}\n")

        #     print(f"\tObject: {obj}")
        #     print(f"\t\tDice: {dice}")
        #     print(f"\t\tdice = (2 * {intersect_volume}) / ({gt_obj_volume} + ({symm_detection_svoxels_volume} / 2))")
        # print("")


    dice_arr = np.array(dice_list)
    dice_mean = np.mean(dice_arr)
    dice_std = np.std(dice_arr)

    print(f"----------------------")
    print(f"# Mean DICE: {dice_mean} +- {dice_std}")

    out_csv.close()

if __name__ == "__main__":
    main()
