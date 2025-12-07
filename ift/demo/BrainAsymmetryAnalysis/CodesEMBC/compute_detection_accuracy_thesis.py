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
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


def main():
    if len(sys.argv) != 5:
        sys.exit(f"python {sys.argv[0]} <result_dir> <hemisphere_mask_dir> <gt_dir> <out_csv_path>\n"
                 "E.g:\n"
                 "result_dir: + thesis_experiments/results/affine_affine/01\n"
                 "                + 000003_000001.nii.gz\n"
                 "                + 000005_000001.nii.gz\n"
                 "                ...\n"
                 "hemisphere_mask_dir: + results/affine_affine/Mask/\n"
                 "                + 000003_000001.nii.gz\n"
                 "                + 000005_000001.nii.gz\n"
                 "                ...\n"
                 "gt_dir: + bases/ATLAS-304/native/labels/primary_stroke\n"
                 "            + 000001_000001.nii.gz\n"
                 "            + 000002_000001.nii.gz\n"
                 "            ...\n\n"
                 "The hemisphere masks must have only two objects: label 1 "
                 "for the right hemisphere, and label 2 for the left one.")

    result_dir = sys.argv[1]
    hem_mask_dir = sys.argv[2]
    gt_dir = sys.argv[3]
    out_csv_path = sys.argv[4]
    validateInputs(result_dir, gt_dir, out_csv_path)

    out_csv = open(out_csv_path, "w")
    out_csv.write("detection_image_path,object,true_positive_rate,num_false_positive_voxels\n")

    n_total_lesions = 0
    n_detected_lesions = 0
    recall_list = []
    n_false_positive_voxels_list = []
    n_false_positive_voxel_rates_list = []

    for detection_img_path in glob.glob(f"{result_dir}/*"):
        image_filename = os.path.basename(detection_img_path)
        hem_mask_path = os.path.join(hem_mask_dir, image_filename)
        gt_path = os.path.join(gt_dir, image_filename)
        print(f"Detection Image: {detection_img_path}")
        print(f"GT: {gt_path}")

        detection_data = ift.ReadImageByExt(detection_img_path).AsNumPy()
        hem_mask_data = ift.ReadImageByExt(hem_mask_path).AsNumPy()
        gt_data = ift.ReadImageByExt(gt_path).AsNumPy()

        hem_vol = hem_mask_data[hem_mask_data != 0].size

        n_objs = gt_data.max()

        for obj in range(1, n_objs + 1):
            n_total_lesions += 1

            gt_obj_bool_data = np.array(gt_data == obj)
            intersec_data = detection_data * gt_obj_bool_data
            intersec_bin_data = intersec_data != 0

            false_positives_data = detection_data - intersec_data
            false_positives_bin_data = false_positives_data != 0

            n_true_positives = np.sum(intersec_bin_data)
            obj_volume = np.sum(gt_obj_bool_data)
            recall = n_true_positives / obj_volume  # true positive rate
            recall_list.append(recall)
            n_false_positive_voxels = np.sum(false_positives_bin_data)
            n_false_positive_voxels_list.append(n_false_positive_voxels)

            n_false_positive_voxel_rate = n_false_positive_voxels / hem_vol
            n_false_positive_voxel_rates_list.append(n_false_positive_voxel_rate)

            out_csv.write(f"{detection_img_path},{obj},{recall},{n_false_positive_voxels}\n")

            print(f"\tObject: {obj}")
            print(f"\t\tNum True Positives: {n_true_positives}")
            print(f"\tHemisphere Volume: {hem_vol}")
            print(f"\t\tObject Volume: {obj_volume}")
            print(f"\t\tRecall: {recall}")
            print(f"\t\tNumber of False Positives: {n_false_positive_voxels}")
            print(f"\t\tNumber of False Positive Rate: {n_false_positive_voxel_rate}")

            if recall > 0.15:
                n_detected_lesions += 1
                print("\t\t### LESION DETECTED!")
        print("")


    perc_detected_lesions = (n_detected_lesions * 1.0) / n_total_lesions

    recall_arr = np.array(recall_list)
    recall_mean = np.mean(recall_arr)
    recall_std = np.std(recall_arr)

    n_false_positive_voxels_arr = np.array(n_false_positive_voxels_list)
    n_false_positive_voxels_mean = np.mean(n_false_positive_voxels_arr)
    n_false_positive_voxels_std = np.std(n_false_positive_voxels_arr)

    n_false_positive_voxel_rates_arr = np.array(n_false_positive_voxel_rates_list)
    n_false_positive_voxel_rates_mean = np.mean(n_false_positive_voxel_rates_arr)
    n_false_positive_voxel_rates_std = np.std(n_false_positive_voxel_rates_arr)

    print(f"----------------------")
    print(f"Num Detected Lesions: {n_detected_lesions}")
    print(f"Total Num Lesions: {n_total_lesions}")
    print(f"# Perc. Detected Lesion: {perc_detected_lesions}")
    print(f"# Mean Recall: {recall_mean} +- {recall_std}")
    print(f"# Mean Number of False Positive Voxels: {n_false_positive_voxels_mean} +- {n_false_positive_voxels_std}")
    print(f"# Mean Number of False Positive Voxel Rate: {n_false_positive_voxel_rates_mean} +- {n_false_positive_voxel_rates_std}")

    out_csv.close()

if __name__ == "__main__":
    main()
