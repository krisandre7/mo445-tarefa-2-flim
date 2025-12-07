import argparse
import os
from os.path import join as pjoin
import sys

import numpy as np
import pyift.pyift as ift


def build_argparse():
    prog_desc = \
        '''
Crop an image according to the bounding box defined by its corresponding mask.
Optionally, resulting cropped image may still be masked so that voxels out of the mask
are ignored in the outputting images. 
'''
    parser = argparse.ArgumentParser(
        description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--image-path', type=str, required=True,
                        help='Image path.')
    parser.add_argument('-m', '--mask-path', type=str, required=True,
                        help='Mask Path.')
    parser.add_argument('-o', '--output-cropped-image-path', type=str, required=True,
                        help='Output path to save the cropped image.')
    parser.add_argument('-y', '--output-cropped-mask-path', type=str, required=False,
                        help='Output path to save the cropped mask.')
    parser.add_argument('-d', '--dilation-radius', type=float, default=0.0,
                        help='Radius used to dilate the mask to then increase '
                             'its corresponding bounding box')
    parser.add_argument('--apply-mask', action='store_true',
                        help='Mask the resulting cropped image by ignoring voxels out the (dilated) mask.')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Image Path: {args.image_path}')
    print(f'- Mask Path: {args.mask_path}')
    print(f'- Output cropped image: {args.output_cropped_image_path}')
    print('--------------------------------------------')
    if args.output_cropped_mask_path:
        print(f'- Output cropped mask: {args.output_cropped_mask_path}')
    print(f'- Dilation Radius: {args.dilation_radius}')
    print(f'- Apply Mask: {args.apply_mask}')
    print('--------------------------------------------\n')


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    filename = os.path.basename(args.image_path)
    print("\t- Reading Image")
    img = ift.ReadImageByExt(args.image_path)
    print("\t- Reading Mask")
    mask = ift.ReadImageByExt(args.mask_path)

    print("\t- Dilating Mask")
    dilated_mask, seeds = ift.DilateBin(mask, None, args.dilation_radius)
    
    if args.apply_mask:
        print("\t- Masking")
        img = ift.Mask(img, dilated_mask)

    print("\t- Min. Bounding Box")
    mbb = ift.MinBoundingBox(dilated_mask, None)
    
    print("\t- Extracting ROI on Image")
    img_roi = ift.ExtractROI(img, mbb)
    img_roi.Write(args.output_cropped_image_path)
    
    if args.output_cropped_mask_path:
        print("\t- Extracting ROI on Mask")
        mask_roi = ift.ExtractROI(mask, mbb)
        mask_roi.Write(args.output_cropped_mask_path)

    print("")



if __name__ == "__main__":
    main()
