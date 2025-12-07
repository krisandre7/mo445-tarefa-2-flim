import argparse
from functools import reduce
import gzip
import os
import sys
import timeit

import numpy as np
from keras.models import load_model
from mykerasmodels import predict

import pyift.pyift as ift


def build_argparse():
    prog_desc = \
        '''
It extracts a feature vector for each supervoxel of a test image by concatening the
histogram of each band inside each supervoxel.
"The filename of each dataset is supervoxel_XXXX.zip, where XXXX is the supervoxel label, starting at 0001.
'''
    parser = argparse.ArgumentParser(
        description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--test-image', type=str, required=True,
                        help='Test Image used to build the supervoxel datasets.')
    parser.add_argument('-s', '--test-supervoxels', type=str, required=True,
                        help='Label Image with the supervoxels for the test image.')
    parser.add_argument('-x', '--training-filtered-set', type=str, required=True,
                        help='Directory or CSV with the pathnames from the training control filtered images.')
    parser.add_argument('-k', '--keras-model', type=str, required=True,
                        help='Keras model (*.h5) with the kernel bank to '
                             'convolve the image set.')
    parser.add_argument('-o', '--out-dir', type=str, required=True,
                        help='Output Directory where the supervoxel datasets will be saved.')
    parser.add_argument('-n', '--bins', type=int, default=128,
                        help='Number of bins of the histogram. Default: 128.')
    parser.add_argument('-m', '--mask', type=str,
                        help='Mask for the input image whose the min. '
                             'bounding box in the objects defines the input '
                             'shape for the convolutions in the keras model')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Test Image Path: {args.test_image}')
    print(f'- Test Supervoxel Path: {args.test_supervoxels}')
    print(f'- Training Filtered Set: {args.training_filtered_set}')
    print(f'- Model: {args.keras_model}')
    print(f'- Output Dir: {args.out_dir}')
    print('--------------------------------------------')
    print(f'- Number of Bins: {args.bins}')
    if args.mask:
        print(f'- Mask: {args.mask}')
    print('--------------------------------------------\n')


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    t1 = timeit.default_timer()

    print('- Loading Keras 3D Convolutional Model')
    model = load_model(args.keras_model)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    train_filt_set = ift.LoadFileSetFromDirOrCSV(
        args.training_filtered_set, 0, True)

    test_img = ift.ReadImageByExt(args.test_image)
    test_svoxels_img = ift.ReadImageByExt(args.test_supervoxels)

    mbb = None
    if args.mask:
            mask = ift.ReadImageByExt(args.mask)
            mbb = ift.MinBoundingBox(mask, None)
            test_data = ift.ExtractROI(test_img, mbb).AsNumPy()
            test_svoxels_img = ift.ExtractROI(test_svoxels_img, mbb)
            test_svoxels_img.Write("svoxels.nii.gz")
    else:
        test_data = test_img.AsNumPy()
    
    t1 = timeit.default_timer()

    print("- Convolving the test image to Kernel Bank")
    test_filt_data = predict(test_data, model)
    test_filt_img = ift.CreateMImageFromNumPy(test_filt_data)

    print("- Building Datasets")
    Zarr = ift.ExtractSupervoxelBandHistFeats(test_filt_img, test_svoxels_img,
                                              train_filt_set, args.bins, None)
    print(f"\tTime: {timeit.default_timer() - t1} secs")
    
    print("- Writing Supervoxel Datasets")
    n_supervoxels = ift.MaximumValue(test_svoxels_img)
    ift.WriteSupervoxelDataSets(Zarr, n_supervoxels, args.out_dir)

    print(f"\nTotal Time: {timeit.default_timer() - t1} secs")
    print("\nDone...")


if __name__ == "__main__":
    main()
