
import os
import sys

import numpy as np
import pyift.pyift as ift


def main():
    if len(sys.argv) != 4:
        sys.exit(f"{sys.argv[0]} <experiment_dir> <gt_dir> <out_pcp.html>")

    experiment_dir = sys.argv[1]
    clf_results_dir = sys.argv[2]
    gt_dir = sys.argv[3]
    out_pcp_path = sys.argv[4]

    print("-----------------------")
    print(f"- Experiment Dir: {experiment_dir}")
    print(f"- Gt Dir: {gt_dir}")
    print(f"- Output PCP: {out_pcp_path}")
    print("-----------------------\n")

    svoxels_path_list = ift.LoadFileSetFromDirOrCSV(svoxels_entry, 0, True).GetPaths()

    volumes_arr = np.zeros(len(svoxels_path_list))
    tpr_arr = np.zeros(len(svoxels_path_list))
    fpr_arr = np.zeros(len(svoxels_path_list))
    got_hit = np.zeros(len(svoxels_path_list), dtype=np.bool)


    for svoxel_path in svoxels_entry:
        print(svoxel_path)




if __name__ == "__main__":
    main()
