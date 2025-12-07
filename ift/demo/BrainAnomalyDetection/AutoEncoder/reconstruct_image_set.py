import argparse
import math
import os
import pdb
import sys
import timeit

from keras.layers import Input
from keras.models import Model, load_model
import numpy as np

import util
import pyift.pyift as ift


def build_argparse():
    prog_desc = \
        '''
Reconstruct an image set from an AutoEncoder3D.
Optionally, you can save the reconstruction errors.
'''
    parser = argparse.ArgumentParser(
        description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--image-set-entry', type=str, required=True,
                        help='CSV or Dir with the images.')
    parser.add_argument('-a', '--autoencoder', type=str, required=True,
                        help='AutoEncoder3D (*.h5).')
    parser.add_argument('-z', '--batch-size', type=int, default=10,
                        help='Batch size for processing. Default: 10')
    parser.add_argument('-o', '--out-reconstructed-images-dir', type=str,
                        required=False, help='Output Dir to save the Reconstructed Images.')
    parser.add_argument('-e', '--out-reconstruction-errors-dir', type=str,
                        required=False, help='Output Dir to save Reconstruction Error Images.')
    parser.add_argument('-l', '--label-image', type=str,
                        help='Label Image with used to mask the output images.')
    parser.add_argument('-b', '--bias', type=str, required=False,
                        help='Pathname from the bias (e.g. normal reconstruction errors) used to attenuate the " \
                             "resulting reconstruction error.')
    parser.add_argument('-g', '--cuda-devices', required=False, type=str,
                        help="Cuda devides to run the script: E.g.: '0,1,2,3' "
                              "to use the Cuda drivers 0, 1, 2, 3")

    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Input Image: {args.image_set_entry}')
    print(f'- AutoEncoder3D: {args.autoencoder}')
    print('--------------------------------------------')
    print(f'- Batch Size: {args.batch_size}')
    if args.out_reconstructed_images_dir:
        print(f'- Output Dir for Reconstructed Image: {args.out_reconstructed_images_dir}')
    if args.out_reconstruction_errors_dir:
        print(f'- Out Dir for Reconstruction Error Image: {args.out_reconstruction_errors_dir}')
    if args.label_image:
        print(f'- Label Image: {args.label_image}')
    if args.bias:
        print(f'- Bias Image: {args.bias}')
    if args.cuda_devices:
        print(f"- Cuda Devices: {args.cuda_devices}")
    print('--------------------------------------------\n')


def validate_args(args):
    if not args.out_reconstructed_images_dir and not args.out_reconstruction_errors_dir:
        sys.exit(f"At least one output directory (reconstruction or error) must be passed")
    
    if not args.autoencoder.endswith('.h5'):
        sys.exit('Invalid AutoEncoder3D extension: %s\nTry *.h5' %
                 args.autoencoder)

    if args.out_reconstructed_images_dir and not os.path.exists(args.out_reconstructed_images_dir):
        os.makedirs(args.out_reconstructed_images_dir)

    if args.out_reconstruction_errors_dir and not os.path.exists(args.out_reconstruction_errors_dir):
        os.makedirs(args.out_reconstruction_errors_dir)


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)
    validate_args(args)

    if args.cuda_devices:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    t1 = timeit.default_timer()

    print('- Loading Image Set Entry')
    img_paths = ift.LoadFileSetFromDirOrCSV(args.image_set_entry, 0, True).GetPaths()
    X = util.read_image_list(img_paths)
    norm_val = util.normalization_value(X)
    X = util.normalize_image_set(X)

    print('- Loading AutoEncoder3D')
    autoencoder = load_model(args.autoencoder, compile=False)
    # print(autoencoder.summary())

    print('- Fitting Image size to the Network\n')
    n_imgs, zsize, ysize, xsize, _ = X.shape
    _, input_zsize, input_ysize, input_xsize, _ = autoencoder.layers[0].output_shape

    X = np.pad(X, ((0, 0), (0, input_zsize - zsize), (0, input_ysize - ysize),
               (0, input_xsize - xsize), (0, 0)), mode="constant")

    output_shape = autoencoder.layers[-1].output_shape  # (batch_size, ...)

    X_rec = np.zeros((n_imgs, *output_shape[1:]), dtype=X.dtype)
    n_iters = math.ceil(n_imgs / args.batch_size)

    for slot in range(n_iters):
        begin = slot * args.batch_size
        end = min(begin + args.batch_size, n_imgs)

        print(f"- Slot 1: [{begin}, {end})")
        print('\t- Reconstructing Image')
        X_rec[begin : end] = autoencoder.predict(X[begin : end])

    print('- Computing Reconstruction Errors\n')

    # (n_imgs, n_voxels) ==> flat
    if X_rec.ndim == 2:
        print('- Reshaping flat reconstructed image')
        n_imgs = X_rec.shape[0]
        X_rec = np.reshape(X_rec, (n_imgs, input_zsize, input_ysize,
                                   input_xsize, 1))
                                   
    X_error = np.abs(X - X_rec)

    mask = np.ones((zsize, ysize, xsize), dtype=np.int32)
    if args.label_image:
        print('- Reading Label Image')
        mask = ift.ReadImageByExt(args.label_image).AsNumPy()

    bias = np.zeros((zsize, ysize, xsize), dtype=np.int32)
    if args.bias:
        bias = ift.ReadImageByExt(args.bias).AsNumPy()

    print('- Saving Images')
    for i, img_path in enumerate(img_paths):
        img_rec = X_rec[i, :zsize, :ysize, :xsize, 0]  # crop the padding
        img_rec = img_rec * norm_val
        img_rec = img_rec.astype(np.int32)
        img_rec[mask == 0] = 0

        filename = os.path.basename(img_path)

        if args.out_reconstructed_images_dir:
            out_rec_img_path = os.path.join(args.out_reconstructed_images_dir,
                                            filename)
            print(f"\tRec. Image: {out_rec_img_path}")
            ift.CreateImageFromNumPy(img_rec, True).Write(out_rec_img_path)

        if args.out_reconstruction_errors_dir:
            img_error = X_error[i, :zsize, :ysize, :xsize, 0]  # crop the padding
            img_error = img_error * norm_val
            img_error = img_error.astype(np.int32)
            img_error = img_error - bias
            img_error[img_error < 0] = 0
            img_error[mask == 0] = 0

            out_rec_error_path = os.path.join(
                args.out_reconstruction_errors_dir, filename)
            print(f"\tRec. Error: {out_rec_error_path}")
            ift.CreateImageFromNumPy(img_error, True).Write(out_rec_error_path)
        print("")


    t2 = timeit.default_timer()

    print(f"\nTotal Time: {t2 - t1} secs")
    print('\n- Done...')


if __name__ == '__main__':
    main()
