import glob
import os
import sys

import numpy as np
import pyift.pyift as ift


def validateInputs(result_dir, svoxels_dir, gt_dir, out_csv_path):
    if not os.path.exists(result_dir):
        sys.exit(f"Result Dir: {result_dir} does not exists")
    if not os.path.exists(svoxels_dir):
        sys.exit(f"Symmetrical Suppervoxels dir Dir: {svoxels_dir} does not exists")
    if not os.path.exists(gt_dir):
        sys.exit(f"GT Dir: {result_dir} does not exists")
    if not out_csv_path.lower().endswith(".csv"):
        sys.exit(f"Invalid CSV file: {out_csv_path}")

    parent_dir = os.path.dirname(out_csv_path)
    if not parent_dir or not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


def main():
    if len(sys.argv) != 5:
        sys.exit(f"python {sys.argv[0]} <result_dir> <svoxels_dir> <gt_dir> <out_csv_path>\n"
                 "E.g:\n"
                 "result_dir: ~/workspace/exps/2020_Martins_RegistrationErrors/baselines/SAAD_ISBI/results_hemisferic_lesions\n"
                 "            - 000001_000001.nii.gz\n"
                 "            - 000002_000001.nii.gz\n"
                 "            ...\n\n"
                 "svoxels_dir: ~/workspace/exps/2020_Martins_RegistrationErrors/baselines/SAAD_ISBI/supervoxels\n"
                 "            - 000001_000001.nii.gz\n"
                 "            - 000002_000001.nii.gz\n"
                 "            ...\n\n"
                 "gt_dir: ~/workspace/exps/2020_Martins_RegistrationErrors/baselines/SAAD_ISBI/gts\n"
                 "            - 000001_000001.nii.gz\n"
                 "            - 000002_000001.nii.gz\n"
                 "            ...\n")

    result_dir = sys.argv[1]
    svoxels_dir = sys.argv[2]
    gt_dir = sys.argv[3]
    out_csv_path = sys.argv[4]
    validateInputs(result_dir, svoxels_dir, gt_dir, out_csv_path)

    detection_img_pathnames = sorted(glob.glob(os.path.join(result_dir, "*")))

    out_csv = open(out_csv_path, "w")
    out_csv.write("detection_image_path,object,num_svoxels,num_detected_svoxels,num_false_positive_svoxels,num_false_positive_connected_svoxels\n")

    n_svoxels_list = []
    n_svoxels_excluding_connected_svoxels_list = []
    n_detected_svoxels_list = []
    n_false_positive_svoxels_list = []
    n_false_positive_connected_svoxels_list = []

    for detection_img_path in detection_img_pathnames:
        filename = os.path.basename(detection_img_path)
        svoxels_img_path = os.path.join(svoxels_dir, filename)
        gt_path = os.path.join(gt_dir, filename)
        print(f"Detection Image: {detection_img_path}")
        print(f"SVoxels Image: {svoxels_img_path}")
        print(f"GT: {gt_path}")

        detection_data = ift.ReadImageByExt(detection_img_path).AsNumPy()
        mid_sagittal_plane = detection_data.shape[2] // 2
        detection_data[:, :, (mid_sagittal_plane + 1):] = 0  # we are also considering the mid-sagittal plane

        # just a test - relabeling the connected detected supervoxels
        # detection_bin_data = detection_data != 0
        # detection_bin_img = ift.CreateImageFromNumPy(detection_bin_data.astype("int32"), True)
        # detection_connected_img = ift.FastLabelComp(detection_bin_img, ift.Spheric(1.74))
        # detection_connected_img.Write(f"tmp/{image_key}.nii.gz")
        # detection_data = detection_connected_img.AsNumPy()

        gt_img = ift.ReadImageByExt(gt_path)
        gt_img_flip = ift.FlipImage(gt_img, ift.IFT_AXIS_X)

        gt_data = gt_img.AsNumPy()
        gt_data_flip = gt_img_flip.AsNumPy()

        n_objs = gt_data.max()

        for obj in range(1, n_objs + 1):
            print(f"\t- Object: {obj}")
            gt_obj_bool_data = np.array(gt_data == obj)
            gt_obj_bool_data_flip = np.array(gt_data_flip == obj)
            gt_obj_bool_data = gt_obj_bool_data + gt_obj_bool_data_flip
            gt_obj_bool_data[:, :, (mid_sagittal_plane + 1):] = 0  # we are also considering the mid-sagittal plane
            n_gt_voxels = np.sum(gt_obj_bool_data)

            intersec_data = detection_data * gt_obj_bool_data
            intersec_svoxels = np.unique(intersec_data)
            nonzero_obj_indices = np.nonzero(intersec_svoxels)[0]
            intersec_svoxels = intersec_svoxels[nonzero_obj_indices]

            # consider all of detected svoxels as false positives initially
            false_positive_svoxels_data = np.array(detection_data)

            # eliminate the true positive svoxels from the false positive data
            for svoxel in intersec_svoxels:
                total_n_obj_voxels = detection_data[detection_data == svoxel].size
                n_intersec_voxels = intersec_data[intersec_data == svoxel].size

                # intersec_perc = (n_intersec_voxels * 1.0) / n_gt_voxels  # just a test - it considers the intersection with respect to the gt volume

                intersec_perc = (n_intersec_voxels * 1.0) / total_n_obj_voxels
                print(f"\t\t- SVoxel: {svoxel} - Perc. of Intersected Voxels: {intersec_perc}")

                # true positive svoxel
                if intersec_perc >= 0.15:
                    false_positive_svoxels_data[false_positive_svoxels_data == svoxel] = 0
                else:
                    print(f"\t\t\t- False Positive")

            n_svoxels = np.max(ift.ReadImageByExt(svoxels_img_path).AsNumPy())
            n_svoxels_list.append(n_svoxels)

            n_detected_svoxels = np.unique(detection_data).size - 1  # excludes the 0-label
            n_detected_svoxels_list.append(n_detected_svoxels)

            n_false_positive_svoxels = np.unique(false_positive_svoxels_data).size - 1  # excludes the 0-label
            n_false_positive_svoxels_list.append(n_false_positive_svoxels)

            false_positive_svoxels_bin_data = false_positive_svoxels_data != 0
            false_positive_svoxels_bin_img = ift.CreateImageFromNumPy(false_positive_svoxels_bin_data.astype("int32"), True)
            false_positive_connected_svoxels_img = ift.FastLabelComp(false_positive_svoxels_bin_img, ift.Spheric(1.74))
            false_positive_connected_svoxels_data = false_positive_connected_svoxels_img.AsNumPy()

            n_false_positive_connected_svoxels = np.unique(false_positive_connected_svoxels_data).size - 1  # excludes the 0-label
            # it seems that, for some cases, ISF is generating some disconnected components with the same label
            # these little components are too small, so I did this trick below to avoid miscomputations
            if n_false_positive_connected_svoxels > n_false_positive_svoxels:
                n_false_positive_connected_svoxels = n_false_positive_svoxels

            n_false_positive_connected_svoxels_list.append(n_false_positive_connected_svoxels)

            n_svoxels_excluding_connected_svoxels_list.append(n_svoxels - n_false_positive_svoxels + n_false_positive_connected_svoxels)

            out_csv.write(f"{detection_img_path},{obj},{n_svoxels},{n_detected_svoxels},{n_false_positive_svoxels},{n_false_positive_connected_svoxels}\n")

            print(f"\tObject: {obj}")
            print(f"\t\tTotal Num of SVoxels: {n_svoxels}")
            print(f"\t\tNum of Detected SVoxels: {n_detected_svoxels}")
            print(f"\t\tNum of False Positive SVoxels: {n_false_positive_svoxels}")
            print(f"\t\tNum of False Positive Connected SVoxels: {n_false_positive_connected_svoxels}")

            # import pdb; pdb.set_trace()


    n_svoxels_arr = np.array(n_svoxels_list)
    n_svoxels_excluding_connected_svoxels_arr = np.array(n_svoxels_excluding_connected_svoxels_list)
    n_detected_svoxels_arr = np.array(n_detected_svoxels_list)
    n_false_positive_svoxels_arr = np.array(n_false_positive_svoxels_list)
    n_false_positive_connected_svoxels_arr = np.array(n_false_positive_connected_svoxels_list)

    n_false_positive_svoxels_on_global_mean = np.mean(n_false_positive_svoxels_arr / n_svoxels_arr)
    n_false_positive_svoxels_on_global_std = np.std(n_false_positive_svoxels_arr / n_svoxels_arr)

    n_false_positive_svoxels_on_detected_mean = np.mean(n_false_positive_svoxels_arr / n_detected_svoxels_arr)
    n_false_positive_svoxels_on_detected_std = np.std(n_false_positive_svoxels_arr / n_detected_svoxels_arr)

    n_false_positive_connected_svoxels_on_global_mean = np.mean(n_false_positive_connected_svoxels_arr / n_svoxels_excluding_connected_svoxels_arr)
    n_false_positive_connected_svoxels_on_global_std = np.std(n_false_positive_connected_svoxels_arr / n_svoxels_excluding_connected_svoxels_arr)

    print(f"----------------------")
    print(f"Mean Total Number of SVoxels: {np.mean(n_svoxels_arr)} ± {np.std(n_svoxels_arr)}")
    print(f"Mean Number of DETECTED SVoxels: {np.mean(n_detected_svoxels_arr)} ± {np.std(n_detected_svoxels_arr)}")
    print(f"# Mean Number of False Positive SVoxels: {np.mean(n_false_positive_svoxels_arr)} ± {np.std(n_false_positive_svoxels_arr)}")
    print(f"# Mean Number of False Positive Connected SVoxels: {np.mean(n_false_positive_connected_svoxels_arr)} ± {np.std(n_false_positive_connected_svoxels_arr)}")
    print(f"## Global False Positive SVoxel Rate: {n_false_positive_svoxels_on_global_mean} ± {n_false_positive_svoxels_on_global_std}")
    print(f"## Detected False Positive SVoxel Rate: {n_false_positive_svoxels_on_detected_mean} ± {n_false_positive_svoxels_on_detected_std}")
    print(f"## Global False Positive Connected SVoxel Rate: {n_false_positive_connected_svoxels_on_global_mean} ± {n_false_positive_connected_svoxels_on_global_std}")

    out_csv.close()

if __name__ == "__main__":
    main()
