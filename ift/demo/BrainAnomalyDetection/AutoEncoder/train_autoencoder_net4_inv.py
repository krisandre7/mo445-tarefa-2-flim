import argparse
import math
import os
import pdb
import sys
import timeit

from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D
from keras.models import Model
import numpy as np

import pyift.pyift as ift

import util




def build_argparse():
    prog_desc = \
'''
Train an AutoEncoder3D to reconstruct a training image set.

The filenames's pattern follows the libIFT convention:
Eg: 000020_00000003.png

where 000020 is the class and 00000003 is the image id.
'''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-t', '--train-set-entry', type=str, required=True,
                        help='Directory or CSV file with pathnames from the SOURCE images.')
    parser.add_argument('-o', '--out-autoencoder', type=str, required=True,
                        help='Output AutoEncoder3D file *.h5.')
    parser.add_argument('-e', '--epochs', type=int, default=500, required=True,
                        help='Number of Epochs used for autoencoder training. Default 500')
    parser.add_argument('-b', '--batch-size', type=int, required=False, default=1,
                        help='Batch size used for autoencoder training. '
                             'Default 1')
    parser.add_argument('-g', '--cuda-devices', required=False, type=str,
                        help="Cuda devides to run the script: E.g.: '0,1,2,3' " \
                              "to use the Cuda drivers 0, 1, 2, 3")
    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Train Set entry: {args.train_set_entry}')
    print(f'- Ouput AutoEncoder3D: {args.out_autoencoder}')
    print('--------------------------------------------')
    print(f'- Epochs: {args.epochs}')
    print(f'- Batch Size: {args.batch_size}')
    if args.cuda_devices:
        print(f"- Cuda Devices: {args.cuda_devices}")
    print('--------------------------------------------\n')


def validate_args(args):
    if not os.path.isdir(args.train_set_entry) and not args.train_set_entry.endswith('.csv'):
        sys.exit('Invalid SOURCE image entry: %s\nTry a CSV or a Directory' % args.train_set_entry)

    if not args.out_autoencoder.endswith('.h5'):
        sys.exit('Invalid AutoEncoder3D extension: %s\nTry *.h5' %
                 args.out_autoencoder)

    parent_dir = os.path.dirname(args.out_autoencoder)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


def train_autoencoder(Xsource, Xtarget, epochs=500, batch_size=1):
    # source: https://blog.keras.io/building-autoencoders-in-keras.html
    # Xsource.shape = (n_imgs, zsize, ysize, xsize, n_channels) dataset of 3D gray/color images
    # Xsource[0].shape = shape = (zsize, ysize, xsize, n_channels) 3D gray/color images
    shape = Xsource[0].shape
    print(f"===> Xsource.shape: {Xsource.shape}")
    print(f"===> shape: {shape}")
    n_channels = shape[-1]

    # for the gray channel
    input_img = Input(shape=shape) # (ysize, xsize, n_channels)

    print('  - Setting up AutoEncoder3D Layers')
    # To train an autoencoder, we must setup its architecture in the same
    # (function) scope, otherwise, it does not work.
    x = Conv3D(4, kernel_size=3, activation='relu', padding='same')(input_img)
    x = MaxPooling3D(pool_size=2, padding='same')(x)
    x = Conv3D(8, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=2, padding='same')(x)
    x = Conv3D(16, kernel_size=3, activation='relu', padding='same')(x)
    # "encoded" is the encoded representation of the input
    encoded = MaxPooling3D(pool_size=2, padding='same')(x)

    x = Conv3D(16, kernel_size=3, activation='relu', padding='same')(encoded)
    x = UpSampling3D(size=2)(x)
    x = Conv3D(8, kernel_size=3, activation='relu', padding='same')(x)
    x = UpSampling3D(size=2)(x)
    x = Conv3D(4, kernel_size=3, activation='relu', padding='same')(x)
    x = UpSampling3D(size=2)(x)
    # "decoded" is the lossy reconstruction of the input
    decoded = Conv3D(n_channels, kernel_size=3, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='nadam', loss='mse')

    print('  - Fitting AutoEncoder3D')
    autoencoder.fit(Xsource, Xtarget,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    verbose=True,
                    validation_data=(None))

    return autoencoder


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)
    validate_args(args)

    if args.cuda_devices:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    t1 = timeit.default_timer()

    print('- Loading Image Sets')
    train_set_paths = ift.LoadFileSetFromDirOrCSV(args.train_set_entry, 0, True).GetPaths()
    train_set_paths = train_set_paths
    print(train_set_paths)

    X = util.read_image_list(train_set_paths)
    X = util.normalize_image_set(X)
    # the target network only reduces the data size by three cubic max pooling of 2
    X = util.adjust_image_domain(X, max_pooling_sizes=[2, 2, 2])

    autoencoder = train_autoencoder(X, X, epochs=args.epochs, batch_size=args.batch_size)

    print('- Saving AutoEncoder3D')
    autoencoder.save(args.out_autoencoder)

    t2 = timeit.default_timer()

    print(f"\nTotal Time: {t2 - t1} secs")

    print('Done...')



if __name__ == '__main__':
    main()


