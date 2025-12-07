import argparse
import os
import pdb
import sys
import timeit

from keras.layers import Input
from keras.models import Model, load_model
import keras.losses

import numpy as np
import pyift.pyift as ift

import util
from loss_functions import weighted_mse


def build_argparse():
    prog_desc = \
        '''
Reconstruct an image from an AutoEncoder3D.
Optionally, you can save the reconstruction error.
'''
    parser = argparse.ArgumentParser(
        description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--image', type=str,
                        required=True, help='Input Image.')
    parser.add_argument('-a', '--autoencoder', type=str, required=True,
                        help='AutoEncoder3D (*.h5).')
    parser.add_argument('-o', '--out-reconstructed-image', type=str,
                        required=True, help='Output Reconstructed Image.')
    parser.add_argument('-e', '--out-reconstruction-error', type=str,
                        required=True, help='Output Reconstruction Error Image.')
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
    print(f'- Input Image: {args.image}')
    print(f'- AutoEncoder3D: {args.autoencoder}')
    print(f'- Output Reconstructed Image: {args.out_reconstructed_image}')
    print(f'- Out Reconstruction Error Image: {args.out_reconstruction_error}')
    print('--------------------------------------------')
    if args.label_image:
        print(f'- Label Image: {args.label_image}')
    if args.bias:
        print(f'- Bias Image: {args.bias}')
    if args.cuda_devices:
        print(f"- Cuda Devices: {args.cuda_devices}")
    print('--------------------------------------------\n')


def validate_args(args):
    if not args.autoencoder.endswith('.h5'):
        sys.exit('Invalid AutoEncoder3D extension: %s\nTry *.h5' %
                 args.autoencoder)

    parent_dir = os.path.dirname(args.out_reconstructed_image)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    if args.out_reconstruction_error:
        parent_dir = os.path.dirname(args.out_reconstruction_error)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir)


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)
    validate_args(args)

    if args.cuda_devices:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    t1 = timeit.default_timer()

    print('- Loading Image')
    img = ift.ReadImageByExt(args.image).AsNumPy()
    norm_val = util.normalization_value(img)
    print(f"norm_val = {norm_val}")

    X = np.array(img)
    new_shape = tuple([1, *X.shape, 1])
    X = np.reshape(X, new_shape)
    X = util.normalize_image_set(X)
    # import pdb; pdb.set_trace()

    print('- Loading AutoEncoder3D')
    # keras.losses.custom_loss = weighted_mse
    autoencoder = load_model(args.autoencoder, compile=False)
    # print(autoencoder.summary())

    print('- Fitting Image size to the Network')
    _, zsize, ysize, xsize, _ = X.shape
    _, input_zsize, input_ysize, input_xsize, _ = autoencoder.layers[0].output_shape

    X = np.pad(X, ((0, 0), (0, input_zsize - zsize), (0, input_ysize - ysize),
                   (0, input_xsize - xsize), (0, 0)), mode="constant")

    print('- Reconstructing Image')
    X_rec = autoencoder.predict(X)

    # (n_imgs, n_voxels) ==> flat
    if X_rec.ndim == 2:
        print('- Reshaping flat reconstructed image')
        n_imgs = X_rec.shape[0]
        X_rec = np.reshape(X_rec, (n_imgs, input_zsize, input_ysize,
                                   input_xsize, 1))

    img_rec = X_rec[0, :zsize, :ysize, :xsize, 0]  # crop the padding
    img_rec = img_rec * norm_val
    img_rec = img_rec.astype(np.int32)

    mask = np.ones((zsize, ysize, xsize), dtype=np.int32)
    if args.label_image:
        print('- Reading Label Image')
        mask = ift.ReadImageByExt(args.label_image).AsNumPy()

    img_rec[mask == 0] = 0

    ift.CreateImageFromNumPy(img_rec, True).Write(args.out_reconstructed_image)

    print('- Reconstruction Errors')
    img_error = np.abs(img - img_rec)

    if args.bias:
        bias = ift.ReadImageByExt(args.bias).AsNumPy()
        img_error = img_error - bias
        img_error[img_error < 0] = 0
    img_error[mask == 0] = 0
    ift.CreateImageFromNumPy(img_error, True).Write(args.out_reconstruction_error)

    t2 = timeit.default_timer()

    print(f"\nTotal Time: {t2 - t1} secs")
    print('\n- Done...')


if __name__ == '__main__':
    main()
