import argparse
import os
import sys

import numpy as np
import pyift.pyift as ift
from scipy import stats


def build_argparse():
    prog_desc = \
'''
Normalize an attention map to generate the importance map (weights).
'''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--attention-map-path', type=str, required=True,
                        help='Pathname with the Attention Map to normalize.')
    parser.add_argument('-l', '--label-image-path', type=str, required=True,
                        help='Pathname with the Label Image of the target objects.')
    parser.add_argument('-o', '--out-importance-map-path', type=str, required=True,
                        help='Pathname for the resulting importance map (*.npy)')
    parser.add_argument('-p', '--outlier-perc', type=float, default=0.0,
                        help='Percentage of voxels with the highest values to be removed. Default: 0.0')
    parser.add_argument('--invert-map', action='store_true', help='Invert the Importance Map')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Attention Map: {args.attention_map_path}')
    print(f'- Label Image: {args.label_image_path}')
    print(f'- Output Importance Map: {args.out_importance_map_path}')
    print('--------------------------------------------')
    print(f'- Outlier Perc: {args.outlier_perc}')
    if args.invert_map:
        print(f"- Invert Importance Map: True")
    print('--------------------------------------------\n')


def validate_args(args):
    parent_dir = os.path.dirname(args.out_importance_map_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    if args.outlier_perc < 0.0 or args.outlier_perc > 1.0:
        sys.exit(f"Invalid Outlier Perc: {args.outlier_perc}. Try [0, 1]")


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)
    validate_args(args)

    print("- Reading Label Image")
    label_img = ift.ReadImageByExt(args.label_image_path)
    label_data = label_img.AsNumPy()

    print("- Reading Attention Map")
    attention_map = ift.ReadImageByExt(args.attention_map_path)
    attention_map_data = attention_map.AsNumPy()
    attention_map_data[label_data == 0] = 0

    max_val = ift.MaximumValueInMask(attention_map, label_img)
    hist = ift.CalcGrayImageHist(attention_map, label_img, max_val + 1, max_val, True)
    acc_hist = ift.CalcAccHist(hist).AsNumPy()

    perc_true_voxels = 1.0 - args.outlier_perc
    outlier_vals = np.where(acc_hist > perc_true_voxels)[0]  # it returns a tuple with a single array
    
    if outlier_vals.size != 0:
        lower_bound_outlier_val = outlier_vals[0]
        attention_map_data[attention_map_data >=
                           lower_bound_outlier_val] = lower_bound_outlier_val - 1

    attention_map_data = attention_map_data / attention_map_data.max()

    if args.invert_map:
        print("- Inverting Map")
        attention_map_data[label_data != 0] = 1.0 - attention_map_data[label_data != 0]

    # imp_map_int = attention_map_data * 4095
    # ift.CreateImageFromNumPy(imp_map_int.astype(np.int32), True).Write("tmp/imp_map_int.nii.gz")

    imp_map = attention_map_data / attention_map_data.sum()

    print("- Writing Importance Map")
    np.save(args.out_importance_map_path, imp_map)



if __name__ == '__main__':
    main()
