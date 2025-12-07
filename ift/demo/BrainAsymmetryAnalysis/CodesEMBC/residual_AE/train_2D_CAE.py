import argparse
import math
import os
import pdb
import sys

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import numpy as np

import pyift.pyift as ift


def build_argparse():
    prog_desc = \
'''
Train an AutoEncoder2D for Anomaly Detection from a set of control images.
Given a set of volumetric control images, the zslice (XY plane) of each image is a sample/2D image for reconstruction.
In order to speed up the process, we only consider the bounding box around the cerebrum, defined by the
cerebrum mask.
'''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('train_control_image_entry', type=str,
                        help='Directory or CSV file with pathnames from the Control Volumetric Images.')
    parser.add_argument('cerebrum_mask_dir', type=str,
                        help='Cerebrum Mask Dir.')
    parser.add_argument('out_autoencoder', type=str,
                        help='Output AutoEncoder2D file *.h5.')
    parser.add_argument('-e', '--epochs', type=int, default=150,
                        help='Number of Epochs used for autoencoder training. Default 150')
    parser.add_argument('-b', '--batch-size', type=int, default=1,
                        help='Batch size used for autoencoder training. '
                             'Default 1')
    return parser


def print_args(args):
    print('--------------------------------------------')
    print('- Control Volumetric image entry: %s' % args.train_control_image_entry)
    print('- Cerebrum Mask Dir: %s' % args.cerebrum_mask_dir)
    print('- Output AutoEncoder2D: %s' % args.out_autoencoder)
    print('--------------------------------------------')
    print('- Epochs: %d' % args.epochs)
    print('- Batch Size: %d' % args.batch_size)
    print('--------------------------------------------\n')


def validate_args(args):
    if not os.path.isdir(args.train_control_image_entry) and not args.train_control_image_entry.endswith('.csv'):
        sys.exit('Invalid Control Volumetric image entry: %s\nTry a CSV or a Directory' % args.train_control_image_entry)

    if not args.out_autoencoder.endswith('.h5'):
        sys.exit('Invalid AutoEncoder2D extension: %s\nTry *.h5' %
                 args.out_autoencoder)

    parent_dir = os.path.dirname(args.out_autoencoder)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


def read_image_list(img_set, mask_dir):
    X = np.empty((0, 256, 256))  # empty array with 0 matrices of shape (256, 256)
    bb_list = []

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

        sx = 256.0 / roi.xsize
        sy = 256.0 / roi.ysize
        sz = 1.0
        roi_interp = ift.Interp(roi, sx, sy, sz)
        X = np.concatenate((X, roi_interp.AsNumPy()), axis=0)  # concatenate the "array" of 2D slices to X
        bb_list.append(bb)

    # here, X has the shape (n_total_of_zslices, 256, 256)
    # adding a new dimension to represent the gray images (n_total_of_zslices, 256, 256, 1)
    new_shape = tuple(list(X.shape) + [1])
    X = np.reshape(X, new_shape)

    return X, bb_list


def normalize_image_set(X, norm_val=4095.0):
    # convert the int images to float and normalize them to [0, 1]
    return (X.astype('float32') / norm_val)


def train_autoencoder(Xsource, Xtarget, epochs=500, batch_size=1):
    # source: https://blog.keras.io/building-autoencoders-in-keras.html
    # Xsource.shape = (n_imgs, ysize, xsize, n_channels) dataset of 2D gray/color images
    # Xsource[0].shape = shape = (ysize, xsize, n_channels) 2D gray/color images
    shape = Xsource[0].shape
    n_channels = shape[-1]

    # for the gray channel
    input_img = Input(shape=shape) # (ysize, xsize, n_channels)

    print('  - Setting up AutoEncoder2D Layers')
    # To train an autoencoder, we must setup its architecture in the same
    # (function) scope, otherwise, it does not work.
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    # "encoded" is the encoded representation of the input
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    # "decoded" is the lossy reconstruction of the input
    decoded = Conv2D(n_channels, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='nadam', loss='mse')

    print('  - Fitting AutoEncoder2D')
    autoencoder.fit(Xsource, Xtarget,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    verbose=1,
                    validation_data=(None))

    return autoencoder



def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)
    validate_args(args)

    train_set = ift.LoadFileSetFromDirOrCSV(args.train_control_image_entry, 0, True).GetPaths()

    print("- Reading Control Images")
    Xtrain, bb_list = read_image_list(train_set, args.cerebrum_mask_dir)
    print("Xtrain.shape: {0}".format(Xtrain.shape))

    # Xtrain.shape ==> (n_total_of_zslices, 256, 256, 1) --- images were cropped
    print("- Normalizing Images")
    norm_val = 4095.0
    Xtrain = normalize_image_set(Xtrain, norm_val)

    print("- Training AutoEncoder")
    autoencoder = train_autoencoder(Xtrain, Xtrain, epochs=args.epochs, batch_size=args.batch_size)

    print('- Saving AutoEncoder2D')
    autoencoder.save(args.out_autoencoder)

    print('Done...')


if __name__ == '__main__':
    main()


