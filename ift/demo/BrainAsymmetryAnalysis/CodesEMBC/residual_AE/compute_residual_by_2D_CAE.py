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
Computes the binary image from the residual between the input images and their reconstructions by 2D CAE.
'''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('test_image_entry', type=str,
                        help='Directory or CSV file with pathnames from the Testing Volumetric Images.')
    parser.add_argument('cerebrum_mask_dir', type=str,
                        help='Directory with the Cerebrum Mask.')
    parser.add_argument('autoencoder', type=str,
                        help='AutoEncoder2D (*.h5).')
    parser.add_argument('output_dir', type=str,
                        help='Output Directory where the binary images from the residuals will be saved.')
    parser.add_argument('-t', '--threshold', type=int, default=500,
                        help='Threshold for binarizing the residuals. Default: 500')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print('- Testing Volumetry Image Entry: %s' % args.test_image_entry)
    print('- Cerebrum Mask Dir: %s' % args.cerebrum_mask_dir)
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


def count_total_of_slices(img_set, mask_dir):
    n_total_slices = 0

    for img_path in img_set:
        filename = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, filename)

        if not os.path.exists(mask_path):
            sys.exit("**ERROR: Mask does not exist\n")

        mask = ift.ReadImageByExt(mask_path)
        bb = ift.MinBoundingBox(mask, None)
        n_total_slices += (bb.end.z - bb.begin.z + 1)

    return n_total_slices


def read_image_list(img_set, mask_dir):
    n_total_slices = count_total_of_slices(img_set, mask_dir)
    print("n_total_slices = %d\n" % n_total_slices)

    X = np.zeros((n_total_slices, 256, 256), dtype=np.int32)  # empty array with 0 matrices of shape (256, 256)
    bb_list = []
    idxs = []  # starting and ending indices in the output array for each 3D image in the set

    last = -1

    for img_path in img_set:
        filename = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, filename)
        print((img_path, mask_path))

        if not os.path.exists(mask_path):
            print("** Mask does not exist\n")
            continue

        mask = ift.ReadImageByExt(mask_path)
        img = ift.Mask(ift.ReadImageByExt(img_path), mask)
        bb = ift.MinBoundingBox(mask, None)
        roi = ift.ExtractROI(img, bb)
        n_slices = roi.zsize

        sx = 256.0 / roi.xsize
        sy = 256.0 / roi.ysize
        sz = 1.0
        roi_interp = ift.Interp(roi, sx, sy, sz)

        start = last + 1
        end = start + n_slices - 1
        last = end
        bb_list.append(bb)
        idxs.append((start, end))

        X[start: end + 1] = roi_interp.AsNumPy()  # inserts the "array" of 2D slices to X


    # here, X has the shape (n_total_of_zslices, 256, 256)
    # adding a new dimension to represent the gray images (n_total_of_zslices, 256, 256, 1)
    new_shape = tuple(list(X.shape) + [1])
    X = np.reshape(X, new_shape)

    return X, bb_list, idxs


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
    X, bb_list, idxs = read_image_list(test_set, args.cerebrum_mask_dir)
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

    for i in range(n_imgs):
        filename = os.path.basename(test_set[i])
        out_bin_path = os.path.join(args.output_dir, "thres_%d" % args.threshold, filename)

        print("[%d/%d] %s" % (i + 1, n_imgs, filename))

        img = ift.ReadImageByExt(test_set[i])

        start, end = idxs[i]
        residual_roi = ift.CreateImageFromNumPy(Xresidual[start: end + 1], True)
        residual_roi.dx, residual_roi.dy, residual_roi.dz = img.dx, img.dy, img.dz

        # roi = ift.CreateImageFromNumPy(X[start: end + 1], True)
        # roi.dx, roi.dy, roi.dz = img.dx, img.dy, img.dz
        # roi.Write("tmp/%s" % filename)
        #
        # rec_roi = ift.CreateImageFromNumPy(Xrec[start: end + 1], True)
        # rec_roi.dx, rec_roi.dy, rec_roi.dz = img.dx, img.dy, img.dz
        # rec_roi.Write("tmp/rec_%s" % filename)


        bin_roi_interp = ift.Threshold(residual_roi, args.threshold, 999999, 1)

        bb = bb_list[i]
        sx = (bb.end.x - bb.begin.x + 1) / bin_roi_interp.xsize
        sy = (bb.end.y - bb.begin.y + 1) / bin_roi_interp.ysize
        sz = 1.0

        bin_roi = ift.InterpByNearestNeighbor(bin_roi_interp, sx, sy, sz)

        bin_img = ift.CreateImageFromImage(img)
        ift.InsertROI(bin_roi, bin_img, bb.begin)
        bin_img.Write(out_bin_path)


if __name__ == '__main__':
    main()



