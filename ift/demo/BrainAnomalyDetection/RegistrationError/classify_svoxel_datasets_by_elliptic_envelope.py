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
    parser.add_argument('-f', '--support-fraction', type=float, default=None,
                        help='The proportion of points [0, 1] to be included in the support of the raw MCD estimate" \
                             "If None, the minimum value of support_fraction will be used within the algorithm: "\
                             "[n_sample + n_features + 1] / 2.')
    parser.add_argument('--normalize-datasets', action='store_true',
                        help='Normalize Datasets')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print('- Supervoxels Path: %s' % args.supervoxels)
    print('- Datasets Entry: %s' % args.supervoxels_datasets_entry)
    print('- Output Anomalous Supervoxels Image: %s' % args.output_anomalous_supervoxels_img)
    print('--------------------------------------------')
    print(f"- Support Fraction: {args.support_fraction}")
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

        Xtrain = X[:-1, :]
        Xtest = X[-1].reshape(1, X.shape[1])

        print("Xtrain.shape: {0}".format(Xtrain.shape))
        print("Xtest.shape: {0}".format(Xtest.shape))

        if args.normalize_datasets:
            print('- Normalization...')
            scaler = MinMaxScaler().fit(Xtrain)
            Xtrain = scaler.transform(Xtrain)
            Xtest = scaler.transform(Xtest)

        print('- Training Elliptic Envelope')
        try:
            clf = EllipticEnvelope(support_fraction=args.support_fraction)
            # clf = IsolationForest(behaviour='new', contamination=0.25)
            clf.fit(Xtrain)

            print('- Classifying Testing Datasets')
            y_pred_test = clf.predict(Xtest) # an array of labels
            y_pred_test = y_pred_test[0]  # just a single test sample
        except Exception as e:
            # some error/exception
            print("######################## Exception")
            print(e)
            y_pred_test = 1  # let's consider the sample as normal
            # import pdb
            # pdb.set_trace()

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
