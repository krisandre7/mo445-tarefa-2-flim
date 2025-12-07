import argparse
import os
import sys
import timeit

import numpy as np
import pyift.pyift as ift
from mykerasmodels import build_simple_conv3D_model


def build_argparse():
    prog_desc = \
'''
Creates a keras model with random kernels (with zero-mean and unit norm) to
perform 3D convolutions.
'''

    parser = argparse.ArgumentParser(description=prog_desc,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--reference-image', type=str, required=True,
                        help='Reference Image whose shape will be the ' \
                             'input shape size of the keras model.')
    parser.add_argument('-o', '--output-model', type=str, required=True,
                        help='Output keras model (*.h5)')
    parser.add_argument('-n', '--n-kernels', type=int, default=32,
                        help='Number of 3D convolutional squared kernels.' \
                             'Default: 32')
    parser.add_argument('-s', '--kernel-size', type=int, default=5,
                        help='Side size of the 3D squared kernels. It must be '
                             'an odd number. Default: 5 ==> '
                             'kernel shape = (5, 5, 5)')
    parser.add_argument('-a', '--activation', type=str, default="linear",
                        help='Activation function applied after convolution: linear, relu. '
                             'Default: linear (no activation)')
    parser.add_argument('-m', '--mask', type=str,
                        help='Mask for the input image whose the min. '
                             'bounding box in the objects defines the input '
                             'shape for the convolutions in the keras model')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Reference Image: {args.reference_image}')
    print(f'- Output Model: {args.output_model}')
    print('--------------------------------------------')
    print(f'- Number of kernels: {args.n_kernels}')
    print(f'- Kernel Size: {args.kernel_size}')
    print(f'- Activation: {args.activation}')
    if args.mask:
        print(f'- Mask: {args.mask}')
    print('--------------------------------------------\n')


def validate_args(args):
    if not args.output_model.endswith(".h5"):
        sys.exit(f'\nERROR: Output model must have the extension .h5')
    if args.kernel_size % 2 == 0:
        sys.exit(f'\nERROR: Kernel size should be an odd number.')



def main():
    parser = build_argparse()
    args = parser.parse_args()
    validate_args(args)
    print_args(args)

    parent_dir = os.path.dirname(args.output_model)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    print('- Building Keras 3D Convolutional Model')
    ref_img_data = ift.ReadImageByExt(args.reference_image).AsNumPy()
    shape = ref_img_data.shape

    if args.mask:
        print('\t- Getting Shape from Min. Bounding Box')
        mask = ift.ReadImageByExt(args.mask)
        bb = ift.MinBoundingBox(mask, None)
        mask_roi_data = ift.ExtractROI(mask, bb).AsNumPy()
        shape = mask_roi_data.shape
    
    print(f'\t- Input Shape: {shape}')
    
    model = build_simple_conv3D_model(shape, args.n_kernels, args.kernel_size,
                                      args.activation)
    
    print('- Saving Model')
    model.save(args.output_model)


if __name__ == "__main__":
    main()
