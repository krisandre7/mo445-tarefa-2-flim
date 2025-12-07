import glob
import os
import sys

import numpy as np
import pyift.pyift as ift


def validate_inputs(result_dir, gt_dir, out_csv_dir):
    if not os.path.exists(result_dir):
        sys.exit(f"Result Dir: {result_dir} does not exists")
    if not os.path.exists(gt_dir):
        sys.exit(f"GT Dir: {result_dir} does not exists")
    if not os.path.exists(out_csv_dir):
        os.makedirs(out_csv_dir)


def main():
    if len(sys.argv) != 4:
        sys.exit(f"python {sys.argv[0]} <result_dir> <gt_dir> <out_csv_dir>\n"
                 "E.g:\n"
                 "result_dir: + exps/03_symmisf_fining_tunning/split_01\n"
                 "                + alpha-0.6_beta-1_thres-0.5\n"
                 "                    - 000001_000001.nii.gz\n"
                 "                    ...\n"
                 "                + alpha-0.6_beta-1_thres-1.0\n"
                "                    ...\n"
                 "gt_dir: + bases/ATLAS-304/3T/regs/nonrigid/labels/primary_stroke\n"
                 "            + 000001_000001.nii.gz\n"
                 "            + 000002_000001.nii.gz\n"
                 "            ...\n")

    result_dir = sys.argv[1]
    gt_dir = sys.argv[2]
    out_csv_dir = sys.argv[3]
    validate_inputs(result_dir, gt_dir, out_csv_dir)

    grid_search = sorted(os.listdir(result_dir))

    mean_highest_iou_list = []
    csv_mean_highest_iou = open(os.path.join(out_csv_dir, "mean_iou.csv"), "w")

    for param_set in grid_search:
        param_set_dir = os.path.join(result_dir, param_set)
        img_set = sorted(glob.glob(f"{param_set_dir}/**/svoxels.nii.gz"))

        print(f"#### ParamSet: {param_set}")

        csv_param_iou = open(os.path.join(out_csv_dir, f"{param_set}.csv"), "w")
        csv_param_iou.write("symmetric_supervoxel_path,object,iou\n")

        highest_obj_iou_list = []

        for symm_svoxels_path in img_set:
            img_id = symm_svoxels_path.split("/svoxels.nii.gz")[0].split("/")[-1]

            gt_path = os.path.join(gt_dir, f"{img_id}.nii.gz")

            print(f"\tSymmetrica Supervoxel Image: {symm_svoxels_path}")
            print(f"\tGT: {gt_path}")

            symm_svoxels_data = ift.ReadImageByExt(symm_svoxels_path).AsNumPy()
            gt_data = ift.ReadImageByExt(gt_path).AsNumPy()

            n_objs = gt_data.max()
            print(f"\tNum Objs: {n_objs}")

            for obj in range(1, n_objs + 1):
                print(f"\t\tObject: {obj}")

                gt_obj_data_bool = np.array(gt_data == obj)
                intersec_gt_symm_svoxels_data = symm_svoxels_data * gt_obj_data_bool
                intersection_svoxels_arr = np.unique(intersec_gt_symm_svoxels_data[intersec_gt_symm_svoxels_data != 0])

                iou_obj_list = []

                for svoxel_label in intersection_svoxels_arr:
                    symm_svoxel_data_bool = symm_svoxels_data == svoxel_label
                    svoxel_volume = np.sum(symm_svoxel_data_bool) / 2  # svoxel_data_bool is symmetrical

                    intersec_gt_symm_svoxel = gt_obj_data_bool * symm_svoxel_data_bool
                    intersect_gt_svoxel_volume = np.sum(intersec_gt_symm_svoxel)

                    union_gt_symm_svoxel = gt_obj_data_bool + symm_svoxel_data_bool
                    union_gt_svoxel_volume = np.sum(union_gt_symm_svoxel) - svoxel_volume  # removing the symmetry of
                                                                                           # svoxels

                    iou = intersect_gt_svoxel_volume / union_gt_svoxel_volume
                    iou_obj_list.append(iou)

                    print(f"\t\t\tSVoxel: {obj}")
                    print(f"\t\t\t\tIoU: {iou}")

                highest_obj_iou = sorted(iou_obj_list, reverse=True)[0]
                highest_obj_iou_list.append(highest_obj_iou)
                csv_param_iou.write(f"{symm_svoxels_path},{obj},{highest_obj_iou}\n")
                print(f"\t\t\t### highest_obj_iou: {highest_obj_iou}")
            print("")

        highest_obj_iou_arr = np.array(highest_obj_iou_list)
        mean_highest_obj_iou = np.mean(highest_obj_iou_arr)

        mean_highest_iou_list.append(mean_highest_obj_iou)
        csv_mean_highest_iou.write(f"{mean_highest_obj_iou},{param_set}\n")
        csv_param_iou.close()
    csv_mean_highest_iou.close()

if __name__ == "__main__":
    main()
