import pdb
import os
import sys

import numpy as np
import pyift.pyift as ift
from MulticoreTSNE import MulticoreTSNE as TSNE



def compute_iou(target_svoxel_data, gt_data):
    intersec = target_svoxel_data * gt_data
    union = target_svoxel_data + gt_data

    return np.count_nonzero(intersec) / np.count_nonzero(union)



def main():
    if len(sys.argv) != 7:
        sys.exit(f"python {sys.argv[0]} <svoxels_img> <result_img> <gt_img> <svoxels_datasets_dir> <tsne_perpexity> <out_dir>")

    svoxels_path = sys.argv[1]
    result_path = sys.argv[2]
    gt_path = sys.argv[3]
    datasets_dir = sys.argv[4]
    perp = int(sys.argv[5])
    out_dir = sys.argv[6]

    print("----------------------------")
    print(f"- SVoxels Image: {svoxels_path}")
    print(f"- Clf Result Image: {result_path}")
    print(f"- GT: {gt_path}")
    print(f"- SVoxel Datasets Dir: {datasets_dir}")
    print(f"- t-SNE perplexity: {perp}")
    print(f"- Out Dir: {out_dir}")
    print("----------------------------\n")


    gt_img = ift.ReadImageByExt(gt_path)
    bb = ift.BoundingBox()
    bb.begin.x = bb.begin.y = bb.begin.z = 0
    bb.end.x, bb.end.y, bb.end.z = gt_img.xsize // 2, gt_img.ysize - 1, gt_img.zsize - 1

    gt_flip_img = ift.FlipImage(gt_img, ift.IFT_AXIS_X)
    gt_final_img = ift.Add(gt_img, gt_flip_img)
    ift.FillBoundingBoxInImage(gt_final_img, bb, 0)
    gt_final_data = gt_final_img.AsNumPy()

    svoxels_img = ift.ReadImageByExt(svoxels_path)
    ift.FillBoundingBoxInImage(svoxels_img, bb, 0)
    svoxels_data = svoxels_img.AsNumPy()

    result_img = ift.ReadImageByExt(result_path)
    ift.FillBoundingBoxInImage(result_img, bb, 0)
    result_data = result_img.AsNumPy()


    # find the svoxels with intersection with the GT
    target_svoxels = np.unique(svoxels_data[gt_final_data != 0])
    print(target_svoxels)

    for tsvoxel in target_svoxels:

        target_svoxel_data = np.multiply(svoxels_data, svoxels_data == tsvoxel)
        iou = compute_iou(target_svoxel_data, gt_final_data)
        if iou >= 0.1:
            print(f"Target SVoxel: {tsvoxel} ===> IoU: {iou}")
            dataset_path = os.path.join(datasets_dir, f"supervoxel_{tsvoxel:04d}.zip")
            out_dataset_path = os.path.join(out_dir, f"supervoxel_{tsvoxel:04d}.zip")
            Z = ift.ReadDataSet(dataset_path)
            X = Z.GetData()
            ref_data = Z.GetRefData()

            tsne = TSNE(n_components=2, n_iter=500, verbose=1, perplexity=perp, n_jobs=8)
            Xproj = tsne.fit_transform(X)
            Xproj = Xproj.astype("float32")

            was_detected = np.any(result_data == tsvoxel)
            print(f"\twas_detected: {was_detected}")

            for r, row in enumerate(ref_data):
                row.append(str(iou))
                row.append(str(int(was_detected)))
                row.append(str(Xproj[r, 0]))
                row.append(str(Xproj[r, 1]))

            # true_labels = Z.GetTrueLabels()
            # groups = Z.GetGroups()
            # Zproj = ift.CreateDataSetFromNumPy(Xproj, true_labels)
            # Zproj.SetGroups(groups)
            # Zproj.SetRefData(ref_data)
            # ift.WriteDataSet(Zproj, out_dataset_path)

            Z.SetRefData(ref_data)
            ift.WriteDataSet(Z, out_dataset_path)



if __name__ == "__main__":
    main()
