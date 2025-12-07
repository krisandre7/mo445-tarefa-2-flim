import argparse
import os
import timeit

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import pyift.pyift as ift



def build_argparse():
    prog_desc = \
        '''
        Classify Supervoxels by OneClass SVM.
        '''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--supervoxels', type=str, required=True, help='Supervoxel Image')
    parser.add_argument('-d', '--supervoxels-datasets-entry', type=str, required=True,
                        help='CSV or Directory with the Supervoxel Datasets')
    parser.add_argument('-o', '--output-anomalous-supervoxels-img', type=str, required=True,
                        help='Output Image with anomalous supervoxels')
    parser.add_argument('-n', '--n_neighbors', type=int, default=20, help='Contamination.')
    parser.add_argument('-c', '--contamination', type=float, default=0.1, help='Contamination.')
    parser.add_argument('--normalize-datasets', action='store_true',
                        help='Normalize Datasets')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print('- Supervoxels Path: %s' % args.supervoxels)
    print('- Datasets Entry: %s' % args.supervoxels_datasets_entry)
    print('- Output Anomalous Supervoxels Image: %s' % args.output_anomalous_supervoxels_img)
    print('--------------------------------------------')
    print(f"- Number of Neighboors: {args.n_neighbors}")
    print(f"- Contamination: {args.contamination}")
    print('--------------------------------------------\n')


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    symm_supervoxels_img = ift.ReadImageByExt(args.supervoxels)
    symm_supervoxels_img_data = symm_supervoxels_img.AsNumPy()
    anomalous_svoxels_data = np.zeros(symm_supervoxels_img_data.shape, dtype=np.int32)

    paths = ift.LoadFileSetFromDirOrCSV(args.supervoxels_datasets_entry, 0, True).GetPaths()

    t1 = timeit.default_timer()

    for dataset_path in paths:
        print("\n\n%s" % dataset_path)
        label = int(os.path.basename(dataset_path).split(".")[0].split("_")[1])
        print(f"label = {label}")
        X = ift.ReadDataSet(dataset_path).GetData()

        print('- Training and Fitting')
        clf = LocalOutlierFactor(
            n_neighbors=args.n_neighbors, contamination=args.contamination)
        y_pred = clf.fit_predict(X)
        y_pred_test = y_pred[-1]  # just a single test sample

        # symmetric supervoxels classified as anomaly
        if y_pred_test == -1:
            anomalous_svoxels_data[symm_supervoxels_img_data == label] = label

    print('- Writing Anomalous SVoxel Image')
    anomalous_svoxels_img = ift.CreateImageFromNumPy(anomalous_svoxels_data, True)
    ift.WriteImageByExt(anomalous_svoxels_img, args.output_anomalous_supervoxels_img)

    t2 = timeit.default_timer()
    print("\n\nProc. Time: %s secs" % (t2 - t1))


if __name__ == '__main__':
    main()
