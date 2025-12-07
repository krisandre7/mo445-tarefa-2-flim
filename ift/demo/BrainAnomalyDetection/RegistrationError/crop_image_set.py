import argparse
import os
from os.path import join as pjoin
import sys

import numpy as np
import pyift.pyift as ift


def build_argparse():
    prog_desc = \
        '''
Crop an image set according to bounding boxes defined by their corresponding masks.
Optionally, resulting cropped images may still be masked so that voxels out of the masks
are ignored in the outputting images. 
'''
    parser = argparse.ArgumentParser(
        description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--image-set-entry', type=str, required=True,
                        help='Directory or CSV file with the image set pathanmes.')
    parser.add_argument('-m', '--mask-directory', type=str, required=True,
                        help='Directory with the mask images, one for each input image. '
                             'PS: Image and Mask file must have the same filename.')
    parser.add_argument('-o', '--output-image-directory', type=str, required=True,
                        help='Output directory to save the cropped images.')
    parser.add_argument('-y', '--output-mask-directory', type=str, required=False,
                        help='Output directory to save the cropped masks.')
    parser.add_argument('-d', '--dilation-radius', type=float, default=0.0,
                        help='Radius used to dilate the masks to then increase '
                             'their corresponding bounding boxes')
    parser.add_argument('--apply-mask', action='store_true',
                        help='Mask the resulting cropped image by ignoring voxels out the (dilated) masks.')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Image Set Entry: {args.image_set_entry}')
    print(f'- Mask Image Dir: {args.mask_directory}')
    print(f'- Output image dir: {args.output_image_directory}')
    print('--------------------------------------------')
    if args.output_mask_directory:
        print(f'- Output mask dir: {args.output_mask_directory}')
    print(f'- Dilation Radius: {args.dilation_radius}')
    print(f'- Apply Mask: {args.apply_mask}')
    print('--------------------------------------------\n')


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    if not os.path.exists(args.output_image_directory):
        os.makedirs(args.output_image_directory)
    if args.output_mask_directory and not os.path.exists(args.output_image_directory):
        os.makedirs(args.output_image_directory)

    img_paths = ift.LoadFileSetFromDirBySuffix(args.image_set_entry, ".nii.gz", 1).GetPaths()

    for idx, img_path in enumerate(img_paths):
        print(f"[{idx}/{len(img_paths)}] = {img_path}")
        filename = os.path.basename(img_path)
        mask_path = pjoin(args.mask_directory, filename)

        if not os.path.exists(mask_path):
            print(f"\tMask '{mask_path}' does not exist... ignoring it")
            continue

        print("\t- Reading Image")
        img = ift.ReadImageByExt(img_path)
        print("\t- Reading Mask")
        mask = ift.ReadImageByExt(mask_path)
        print("\t- Binarizing Mask")
        bin_mask = ift.Binarize(mask)

        if args.dilation_radius:
            print("\t- Dilating Binarized Mask")
            bin_mask, seeds = ift.DilateBin(bin_mask, None, args.dilation_radius)
        
        if args.apply_mask:
            print("\t- Masking")
            img = ift.Mask(img, mask)

        print("\t- Min. Bounding Box")
        mbb = ift.MinBoundingBox(bin_mask, None)
        
        print("\t- Extracting ROI on Image")
        img_roi = ift.ExtractROI(img, mbb)
        out_img_path = pjoin(args.output_image_directory, filename)
        img_roi.Write(out_img_path)
        
        if args.output_mask_directory:
            print("\t- Extracting ROI on Mask")
            mask_roi = ift.ExtractROI(mask, mbb)
            out_mask_path = pjoin(args.output_mask_directory, filename)
            mask_roi.Write(out_mask_path)

        print("")



if __name__ == "__main__":
    main()
