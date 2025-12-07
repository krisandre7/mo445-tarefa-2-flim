import argparse
import os
import timeit

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm

import pyift.pyift as ift


def build_argparse():
    prog_desc = \
        '''
        Classifies brain asymmetries by OneClass SVM.
        '''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('symmetric_supervoxels_path', type=str, help='Supervoxel Image')
    parser.add_argument('supervoxels_datasets_entry', type=str, help='CSV or Directory with the Symmetric Supervoxel Datasets')
    parser.add_argument('output_anomalous_supervoxels_img', type=str, help='Output Image with anomalous symmetric supervoxels')
    parser.add_argument('-k', '--kernel', type=str, default='linear', help='SVM kernel: linear or rbf. Default: linear')
    parser.add_argument('-n', '--nu', type=float, default=0.1, help='NU parameter. Default: 0.1')
    parser.add_argument('-g', '--gamma', type=float, default=0.01, help='GAMMA parameter. Default: 0.01')
    parser.add_argument('--normalize-datasets', action='store_true',
                        help='Normalize Datasets')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print('- Supervoxels Path: %s' % args.symmetric_supervoxels_path)
    print('- Datasets Entry: %s' % args.supervoxels_datasets_entry)
    print('- Output Anomalous Supervoxels Image: %s' % args.output_anomalous_supervoxels_img)
    print('--------------------------------------------')
    print('- Kernel: %s' % args.kernel)
    print('- NU: %f' % args.nu)
    print('- GAMMA: %f' % args.gamma)
    print('--------------------------------------------\n')


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    symm_supervoxels_img = ift.ReadImageByExt(args.symmetric_supervoxels_path)
    symm_supervoxels_img_data = symm_supervoxels_img.AsNumPy()
    asym_img_data = np.zeros(symm_supervoxels_img_data.shape, dtype=np.int32)

    paths = ift.LoadFileSetFromDirOrCSV(args.supervoxels_datasets_entry, 0, True).GetPaths()

    t1 = timeit.default_timer()

    for dataset_path in paths:
        print("\n\n%s" % dataset_path)
        label = int(os.path.basename(dataset_path).split(".")[0].split("_")[1])
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

        print('- Training One Class SVM')
        try:
            clf = svm.OneClassSVM(kernel=args.kernel, nu=args.nu, gamma=args.gamma, verbose=True)
            clf.fit(Xtrain)

            print('- Classifying Testing Datasets')
            y_pred_test = clf.predict(Xtest)
        except Exception as e:
            print("Exception")
            print(e)
            import pdb
            pdb.set_trace()

        y_pred_test = y_pred_test[0]  # just a single test sample

        # symmetric supervoxels classified as anomaly
        if y_pred_test == -1:
            asym_img_data[symm_supervoxels_img_data == label] = label

    print('- Writing Asymmetry Image')
    asym_img = ift.CreateImageFromNumPy(asym_img_data, True)
    ift.WriteImageByExt(asym_img, args.output_anomalous_supervoxels_img)

    t2 = timeit.default_timer()
    print("\n\nProc. Time: %s secs" % (t2 - t1))


if __name__ == '__main__':
    main()
