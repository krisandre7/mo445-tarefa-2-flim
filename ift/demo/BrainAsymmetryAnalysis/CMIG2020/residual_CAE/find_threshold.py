import argparse
import math
import os
import pdb
import sys

from keras.models import load_model
import numpy as np

import pyift.pyift as ift


def build_argparse():
    prog_desc = \
'''
Finds the "best" threshold for the 2D CAE.
It consists a given percentile of the histogram of all reconstruction errors of the training images.
In order to speed up the process, consider to pass cropped images around the cerebrum.
'''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-t', '--train-control-image-entry', type=str, required=True,
                        help='Directory or CSV file with pathnames from the Control Volumetric Images. '
                             'All images must have the same size. Use the same training images used to train the autoencoder.')
    parser.add_argument('-a', '--autoencoder', required=True, type=str,
                        help='AutoEncoder2D (*.h5).')
    parser.add_argument('-p', '--percentile', type=float, help='Percentile of Histogram [0, 1.0]. Default= 98.0', default=0.98)

    return parser


def print_args(args):
    print('--------------------------------------------')
    print('- Control Volumetric image entry: %s' % args.train_control_image_entry)
    print('- AutoEncoder2D: %s' % args.autoencoder)
    print('- Percentile: %s' % args.percentile)
    print('--------------------------------------------\n')


def validate_args(args):
    if not os.path.isdir(args.train_control_image_entry) and not args.train_control_image_entry.endswith('.csv'):
        sys.exit('Invalid Control Volumetric image entry: %s\nTry a CSV or a Directory' % args.train_control_image_entry)

    if not args.autoencoder.endswith('.h5'):
        sys.exit('Invalid AutoEncoder2D extension: %s\nTry *.h5' %
                 args.autoencoder)

    if args.percentile < 0.0 or args.percentile > 1.0:
        sys.exit("Invalid percentile: %f... Try [0, 1.0]" % args.percentile)


def filter_train_set(img_set, mask_dir):
    filt_img_set = []

    for img_path in img_set:
        filename = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, filename)

        if os.path.exists(mask_path):
            filt_img_set.append(img_path)

    return filt_img_set


def count_total_of_slices(img_set, mask_dir):
    n_total_slices = 0

    for img_path in img_set:
        filename = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, filename)

        if not os.path.exists(mask_path):
            continue

        mask = ift.ReadImageByExt(mask_path)
        bb = ift.MinBoundingBox(mask, None)
        n_total_slices += (bb.end.z - bb.begin.z + 1)

    return n_total_slices


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

    train_set = ift.LoadFileSetFromDirOrCSV(args.train_control_image_entry, 0, True).GetPaths()

    print("- Reading Control Images")
    X = read_image_list(train_set)

    shape_without_channels = tuple(list(X.shape)[:-1])
    print("\tX.shape: {0}".format(X.shape))
    print("\tshape_without_channels: {0}".format(shape_without_channels))

    print('- Loading AutoEncoder2D')
    autoencoder = load_model(args.autoencoder)

    # X.shape ==> (n_total_of_zslices, 256, 256, 1) --- images were cropped
    print("- Normalizing Images")
    norm_val = 4095
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

    print("\t- Getting Threshold")
    hist = np.histogram(Xresidual, bins=(norm_val + 1))[0]
    acc_hist = np.array(hist, dtype=float)

    for bin in range(1, norm_val + 1):
        acc_hist[bin] = acc_hist[bin - 1] + hist[bin]

    # np.save("hist_AE.npy", hist)
    np.save("acc_hist_AE.npy", acc_hist)
    sum = float(acc_hist[-1])

    threshold = 0

    for bin in range(norm_val + 1):
        acc_hist[bin] = acc_val = acc_hist[bin] / sum

        if acc_val >= args.percentile:
            threshold = bin
            break

    print("\n********")
    print("- Found Threshold for %f percentile: %d" % (args.percentile, threshold))
    print("********\n")


if __name__ == '__main__':
    main()



