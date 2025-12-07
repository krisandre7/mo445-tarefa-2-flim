import argparse
import math
import os
import pdb
import sys
import timeit

from keras.models import load_model
import numpy as np

import pyift.pyift as ift


def build_argparse():
    prog_desc = \
'''
Computes the binary image from the residual between the input images and their reconstructions by 2D CAE.
In order to speed up the process, consider to pass cropped images around the cerebrum.
'''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--test-image-entry', type=str, required=True,
                        help='Directory or CSV file with pathnames from the Testing Volumetric Images. '
                             'All images must have the same size.')
    parser.add_argument('-a', '--autoencoder', type=str, required=True,
                        help='AutoEncoder2D (*.h5).')

    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='Output Directory where the binary images from the residuals will be saved.')
    parser.add_argument('-t', '--threshold', type=int, default=500,
                        help='Threshold for binarizing the residuals. Default: 500')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print('- Testing Volumetry Image Entry: %s' % args.test_image_entry)
    print('- AutoEncoder2D: %s' % args.autoencoder)
    print('- Output Directory: %s' % args.output_dir)
    print('--------------------------------------------')
    print('- Threshold: %d' % args.threshold)
    print('--------------------------------------------\n')


def validate_args(args):
    if not os.path.isdir(args.test_image_entry) and not args.test_image_entry.endswith('.csv'):
        sys.exit('Invalid Testing Volumetric image entry: %s\nTry a CSV or a Directory' % args.test_image_entry)

    if not args.autoencoder.endswith('.h5'):
        sys.exit('Invalid AutoEncoder2D extension: %s\nTry *.h5' %
                 args.autoencoder)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


def read_image_list(img_set):
    img0 = ift.ReadImageByExt(img_set[0])
    n_slices_per_image = img0.zsize
    n_total_slices = len(img_set) * n_slices_per_image

    # empty array with 0 matrices of shape (256, 256)
    X = np.empty((n_total_slices, 256, 256))

    for i, img_path in enumerate(img_set):
        print(f"[{i} / {len(img_set) - 1}] = {img_path}")
        img = ift.ReadImageByExt(img_path)

        start = i * n_slices_per_image
        end = start + n_slices_per_image

        sx = 256.0 / img.xsize
        sy = 256.0 / img.ysize
        sz = 1.0
        img_interp = ift.Interp(img, sx, sy, sz)
        # concatenate the "array" of 2D slices to X
        X[start: end] = img_interp.AsNumPy()

    # here, X has the shape (n_total_of_zslices, 256, 256)
    # adding a new dimension to represent the gray images (n_total_of_zslices, 256, 256, 1)
    new_shape = tuple(list(X.shape) + [1])
    X = np.reshape(X, new_shape)

    return X


def normalize_image_set(X, norm_val=4095.0):
    # convert the int images to float and normalize them to [0, 1]
    return (X.astype('float32') / norm_val)


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)
    validate_args(args)

    test_set = ift.LoadFileSetFromDirOrCSV(args.test_image_entry, 0, True).GetPaths()
    n_imgs = len(test_set)

    print("- Reading Control Images")
    X = read_image_list(test_set)
    n_slices = X.shape[0] // n_imgs

    shape_without_channels = tuple(list(X.shape)[:-1])
    print("\tX.shape: {0}".format(X.shape))
    print("\tshape_without_channels: {0}".format(shape_without_channels))

    print('- Loading AutoEncoder2D')
    autoencoder = load_model(args.autoencoder)

    # X.shape ==> (n_total_of_zslices, 256, 256, 1) --- images were cropped
    print("- Normalizing Images")
    norm_val = 4095.0
    X = normalize_image_set(X, norm_val)

    print("\t- Reconstructing the Images")
    Xrec = autoencoder.predict(X)

    X = X * norm_val
    X = X.astype("int32")
    X = X.reshape(shape_without_channels)

    Xrec = Xrec * norm_val
    Xrec = Xrec.astype("int32")
    Xrec = Xrec.reshape(shape_without_channels)

    print("\t- Getting the Residuals")
    Xresidual = np.absolute(X - Xrec)  # shape (n_total_of_zslices, 256, 256, 1)
    Xresidual = Xresidual.astype("int32")
    Xresidual = Xresidual.reshape(shape_without_channels)

    time_list = []

    for i, img_path in enumerate(test_set):
        t1 = timeit.default_timer()

        filename = os.path.basename(img_path) # ex: 000001_000002.nii.gz
        image_key = filename.split(".nii.gz")[0]

        out_rec_path = os.path.join(args.output_dir, image_key, "reconstruction.nii.gz")
        out_error_path = os.path.join(args.output_dir, image_key, "error.nii.gz")
        out_bin_path = os.path.join(args.output_dir, image_key, "result.nii.gz")

        print("[%d/%d] %s" % (i + 1, n_imgs, filename))

        img = ift.ReadImageByExt(img_path)
        sx = img.xsize / Xrec.shape[2]
        sy = img.ysize / Xrec.shape[1]
        sz = 1.0

        start = i * n_slices
        end = start + n_slices

        rec_roi = ift.CreateImageFromNumPy(Xrec[start: end], True)
        rec_img = ift.Interp(rec_roi, sx, sy, sz)
        rec_img.dx, rec_img.dy, rec_img.dz = img.dx, img.dy, img.dz
        rec_img.Write(out_rec_path)

        rec_error_roi = ift.CreateImageFromNumPy(Xresidual[start: end], True)
        rec_error_img = ift.Interp(rec_error_roi, sx, sy, sz)
        rec_error_img.dx, rec_error_img.dy, rec_error_img.dz = img.dx, img.dy, img.dz
        rec_error_img.Write(out_error_path)

        residual_roi = ift.CreateImageFromNumPy(Xresidual[start : end], True)
        bin_roi = ift.Threshold(residual_roi, args.threshold, 999999, 1)
        bin_img = ift.InterpByNearestNeighbor(bin_roi, sx, sy, sz)
        bin_img.dx, bin_img.dy, bin_img.dz = img.dx, img.dy, img.dz
        bin_img.Write(out_bin_path)

        t2 = timeit.default_timer()
        time_list.append(t2 - t1)
    
    time_list = np.array(time_list)
    print(f"\n\nProc. Time: {np.mean(time_list)} +- {np.std(time_list)} secs")




if __name__ == '__main__':
    main()



