import argparse
import math
import os
import pdb
import sys
import timeit

import keras.backend as K
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Reshape
from keras.models import Sequential
import numpy as np

import pyift.pyift as ift

import util
from loss_functions import weighted_mse




def build_argparse():
    prog_desc = \
'''
Train an AutoEncoder3D to reconstruct a training image set.

It uses the EDT to weight the voxels in AE's loss function (mean square error).
'''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-t', '--train-set-entry', type=str, required=True,
                        help='Directory or CSV file with pathnames from the SOURCE images.')
    parser.add_argument('-w', '--weights-path', type=str, required=True,
                        help='Weights (Importance Map used to weight the reconstruction erros in the loss function (*.npy).')
    parser.add_argument('-o', '--out-autoencoder', type=str, required=True,
                        help='Output AutoEncoder3D file *.h5.')
    parser.add_argument('-e', '--epochs', type=int, default=50, required=True,
                        help='Number of Epochs used for autoencoder training. Default 50')
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
    print(f'- Weights (Importance Map): {args.weights_path}')
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

    if not args.weights_path.endswith('.npy'):
        sys.exit('Invalid Weights Path extension: %s\nTry *.npy' %
                 args.weights_path)

    if not args.out_autoencoder.endswith('.h5'):
        sys.exit('Invalid AutoEncoder3D extension: %s\nTry *.h5' %
                 args.out_autoencoder)

    parent_dir = os.path.dirname(args.out_autoencoder)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)






def train_autoencoder(Xsource, Xtarget, weights, epochs=50, batch_size=1):
    # source: https://blog.keras.io/building-autoencoders-in-keras.html
    # Xsource.shape = (n_imgs, zsize, ysize, xsize, n_channels) dataset of 3D gray/color images
    # Xsource[0].shape = shape = (zsize, ysize, xsize, n_channels) 3D gray/color images
    n_imgs = Xsource.shape[0]
    n_voxels = Xsource[0].size
    shape = Xsource[0].shape
    print(f"===> Xsource.shape: {Xsource.shape}")
    print(f"===> shape: {shape}")
    n_channels = shape[-1]

    Xtarget = np.reshape(Xtarget, (n_imgs, n_voxels))

    weights_flat = weights.flatten()
    weights_flat_rep = np.tile(weights_flat, (batch_size, 1))
    weights_tensor = K.constant(weights_flat_rep)

    # for the gray channel
    print('\t- Setting up AutoEncoder3D Layers')
    autoencoder = Sequential()
    autoencoder.add(Conv3D(32, kernel_size=3, activation='relu', padding='same', input_shape=shape))
    autoencoder.add(MaxPooling3D(pool_size=2, padding='same'))
    autoencoder.add(Conv3D(16, kernel_size=3, activation='relu', padding='same'))
    autoencoder.add(MaxPooling3D(pool_size=2, padding='same'))
    autoencoder.add(Conv3D(8, kernel_size=3, activation='relu', padding='same'))
    # "encoded" is the encoded representation of the input
    autoencoder.add(MaxPooling3D(pool_size=2, padding='same'))

    autoencoder.add(Conv3D(8, kernel_size=3, activation='relu', padding='same'))
    autoencoder.add(UpSampling3D(size=2))
    autoencoder.add(Conv3D(16, kernel_size=3, activation='relu', padding='same'))
    autoencoder.add(UpSampling3D(size=2))
    autoencoder.add(Conv3D(32, kernel_size=3, activation='relu', padding='same'))
    autoencoder.add(UpSampling3D(size=2))
    # "decoded" is the lossy reconstruction of the input
    autoencoder.add(Conv3D(n_channels, kernel_size=3, activation='sigmoid', padding='same'))
    autoencoder.add(Reshape((n_voxels,)))


    autoencoder.compile(optimizer='nadam',
                        loss=weighted_mse(weights_tensor))

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

    X = util.read_image_list(train_set_paths)
    img_shape = X[0].shape
    X = util.normalize_image_set(X)
    X = util.adjust_image_domain(X, max_pooling_sizes=[2, 2, 2])
    _, input_zsize, input_ysize, input_xsize, _ = X.shape

    weights = np.load(args.weights_path)
    weights = np.reshape(weights, (*weights.shape, 1))
    
    if weights.shape != img_shape:
        sys.exit("ERROR: Weights (Importance Map) has different shape from the training images: "
                 f"{weights.shape} != {img_shape}")

    zsize, ysize, xsize, _ = weights.shape
    weights = np.pad(weights, ((0, input_zsize - zsize), (0, input_ysize - ysize),
                              (0, input_xsize - xsize), (0, 0)), mode="constant")

    autoencoder = train_autoencoder(X, X, weights, epochs=args.epochs,
                                    batch_size=args.batch_size)

    print('- Saving AutoEncoder3D')
    autoencoder.save(args.out_autoencoder)

    t2 = timeit.default_timer()

    print(f"\nTotal Time: {t2 - t1} secs")

    print('Done...')



if __name__ == '__main__':
    main()


