import json
import os
from os.path import join as pjoin
from os.path import join as pjoin
import pdb
import shutil
import tempfile

import numpy as np
import pyift.pyift as ift
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines


tmp_dir = tempfile.mkdtemp(prefix="workdir_", dir="/tmp")



def onpick(event, X, ref_data, ax3):
    print("*******")
    idx_list = event.ind
    Xmean = np.mean(X[idx_list], axis=0)
    nbins = Xmean.shape[0]
    ax3.clear()
    ax3.bar(list(range(nbins)), Xmean, color="black")
    ax3.title.set_text('Avg Hist for Selected Sampled')
    ax3.figure.canvas.draw()

    for idx in idx_list:
        print(ref_data[idx])
        img_path = ref_data[idx][0]
        # svoxels_path = ref_data[idx][1]
        svoxels_path = ref_data[idx][1].replace("exps", "exps/SAAD")
        target_svoxel = int(ref_data[idx][2])
        tmp_img_path = tempfile.mkstemp(prefix=f"svoxels_{target_svoxel}_", suffix=".nii.gz", dir=tmp_dir)[1]

        print(img_path)
        print(svoxels_path)
        print(target_svoxel)
        print(tmp_img_path)
        ift.ExtractObject(ift.ReadImageByExt(svoxels_path), target_svoxel).Write(tmp_img_path)
        os.system(f"itksnap -g {img_path} -s {tmp_img_path}")



def main():
    work_dir = "/Users/hisamuka/workspace/exps/UnderstandingWhyLowSegmentation"
    saad_result_dir = pjoin(work_dir, "exps/SAAD")
    gt_dir = pjoin(work_dir, "bases/ATLAS-304/3T/regs/nonrigid/labels/primary_stroke")
    exp_dir = pjoin(work_dir, "exps", "PoorClassification")
    exp_datasets_dir = pjoin(exp_dir, "datasets")

    ##### PREPARING/PROCESSING SVOXELS DATASETS
    # img_id_list = ["000021_000001", "000096_000001", "000135_000001", "000209_000001"]
    img_id_list = []
    img_id_list += ["000131_000001", "000135_000001", "000242_000001"]  # asymmetric lesions with high intersection with svoxels
    img_id_list += ["000096_000001"]  # large asymmetric lesion
    img_id_list += ["000021_000001"]  # subtle lesion but with high intersection with svoxel
    img_id_list += ["000082_000001"]  # large lesion on cortex
    img_id_list += ["000035_000001", "000038_000001", "000157_000001"]  # low asymmetries, poor svoxels
    img_id_list += ["000208_000001"]  # lesion on cortex, low asymmetries, poor svoxels



    # project feats by t-SNE
    # for img_id in img_id_list:
    #     svoxels_path = os.path.join(saad_result_dir, img_id, "svoxels.nii.gz")
    #     clf_result_path = os.path.join(saad_result_dir, img_id, "result.nii.gz")
    #     gt_path = os.path.join(gt_dir, img_id + ".nii.gz")
    #     datasets_dir = os.path.join(saad_result_dir, img_id, "datasets")
    #     out_dir = os.path.join(exp_datasets_dir, img_id)
    #
    #     os.system(f"python $NEWIFT_DIR/demo/BrainAsymmetryAnalysis/EveryDayExps/UnderstandingWhyLowSegmAcc/PoorClassifiers/prepare_svoxel_datasets.py "
    #               f"{svoxels_path} {clf_result_path} {gt_path} {datasets_dir} 40 {out_dir}")


    ##### PLOTING SVOXELS DATASET
    for img_id in sorted(os.listdir(exp_datasets_dir)):
        datasets_img_dir = os.path.join(exp_datasets_dir, img_id)

        for dataset_filename in sorted(os.listdir(datasets_img_dir)):
            svoxel = int(dataset_filename.split(".zip")[0].split("_")[1])
            dataset_path = os.path.join(datasets_img_dir, dataset_filename)
            Z = ift.ReadDataSet(dataset_path)
            X = Z.GetData()
            ref_data = Z.GetRefData()

            Xproj = np.array([[float(row[-2]), float(row[-1])] for row in ref_data])

            Xtrain = X[:-1,:]
            Xproj_train = Xproj[:-1,:]
            ref_data_train = ref_data[:-1]

            Xtest = np.reshape(X[-1], (1, X.shape[1]))
            Xproj_test = np.reshape(Xproj[-1], (1, Xproj.shape[1]))
            ref_data_test = [ref_data[-1]]

            ious = [float(ref_data_test[0][3])]
            print(ious)
            was_detected = bool(int(ref_data_test[0][4]))
            if was_detected:
                test_label = "Test Detected"
                marker = "^"
                cmap = "summer"
                legend_color = "green"
            else:
                test_label = "Test Undetected"
                marker = "X"
                cmap = "autumn"
                legend_color = "red"

            plt.figure(figsize=(16, 8))

            ax1 = plt.subplot(2, 1, 2)
            scatter_train = ax1.scatter(Xproj_train[:, 0], Xproj_train[:, 1], c="black", alpha=0.5, label='Train', marker="o", picker=5)
            scatter_test = ax1.scatter(Xproj_test[:, 0], Xproj_test[:, 1], alpha=0.9, label=test_label, marker=marker, s=100,
                              c=ious, cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=1.0), picker=5)
            plt.colorbar(scatter_test, label="IoU test")

            # trick to know whats is the scatter plot which whose element was pickked
            scatters = {
                scatter_train: "scatter_train",
                scatter_test: "scatter_test"
            }

            train_legend = mlines.Line2D([], [], color='black', marker="o", linestyle='None', markersize=10,
                                         label='Train')
            test_legend = mlines.Line2D([], [], color=legend_color, marker=marker, linestyle='None', markersize=10,
                                        label=test_label)
            ax1.legend(handles=[train_legend, test_legend], loc='best')
            ax1.title.set_text('t-SNE projection')


            ax2 = plt.subplot(2, 2, 1)
            nbins = Xtrain.shape[1]
            print(nbins, range(nbins))
            ax2.bar(list(range(nbins)), np.mean(Xtrain, axis=0), color="black")
            ax2.bar(list(range(nbins)), Xtest[0], color=legend_color)
            ax2.title.set_text('Hist. of Test Feats and Avg Hist for Training Feats')


            ax3 = plt.subplot(2, 2, 2, sharey=ax2)
            ax3.title.set_text('Avg Hist for Selected Sampled')


            plt.gcf().suptitle(f"Image {img_id}, SVoxel: {svoxel}")

            ax1.figure.canvas.mpl_connect('pick_event', lambda event: onpick(event, Xtrain, ref_data_train, ax3) if scatters[event.artist] == "scatter_train" else onpick(event, Xtest, ref_data_test, ax3))
            plt.subplots_adjust(hspace=0.5)
            plt.show()

    shutil.rmtree(tmp_dir)



if __name__ == "__main__":
    main()