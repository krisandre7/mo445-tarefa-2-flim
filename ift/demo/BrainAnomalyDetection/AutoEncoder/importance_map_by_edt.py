import argparse
import os
import sys

import numpy as np
import pyift.pyift as ift


def build_argparse():
    prog_desc = \
'''
Computes an importance map (weights) based on the Euclidean Distance Transform.
'''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-l', '--label-image-path', type=str, required=True,
                        help='Pathname with the Label Image of the target objects.')
    parser.add_argument('-o', '--out-importance-map-path', type=str, required=True,
                        help='Pathname for the resulting importance map (*.npy)')
    parser.add_argument('-f', '--attenuation-function', type=str, default="linear",
                        help='Attenuation Function: [none, linear, exp]. Default: linear')
    parser.add_argument('-a', '--max-attenuation-factor', type=float, default=0.75,
                        help='Maximum Attenuate Factor used to attenuate the errors based on the EDT. Default: 0.75. '\
                             'Not used for \'none\' function')
    parser.add_argument('-x', '--exponent', type=float, default=2.0,
                        help='Exponent for the exponent attenuation function. Default: 2.0. Not used for linear function.')
    parser.add_argument('--invert-map', action='store_true', help='Invert the Importance Map')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Label Image: {args.label_image_path}')
    print(f'- Ouput Importance Map: {args.out_importance_map_path}')
    print('--------------------------------------------')
    print(f"- Attenuation Function: {args.attenuation_function}")
    if args.attenuation_function != "none":
        print(f"- Max Attenuation Factor: {args.max_attenuation_factor}")
        print(f"- Expoent: {args.exponent}")
    if args.invert_map:
        print(f"- Invert Importance Map: True")
    print('--------------------------------------------\n')


def validate_args(args):
    if not args.out_importance_map_path.endswith('.npy'):
        sys.exit('Invalid Importance Map extension: %s\nTry *.npy' %
                 args.out_importance_map_path)

    parent_dir = os.path.dirname(args.out_importance_map_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    if args.attenuation_function not in ["none", "linear", "exp"]:
        sys.exit(f"Invalid Attenuation function: {args.attenuation_function}")

    if args.max_attenuation_factor < 0.0:
        sys.exit(
            f"Invalid Max. Attenuation Factor: {args.max_attenuation_factor}")

    if args.attenuation_function == "exp" and args.exponent <= 0.0:
        sys.exit(f"Invalid Expoent: {args.exponent}")





def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)
    validate_args(args)

    print("- Reading Label Image")
    label_img = ift.ReadImageByExt(args.label_image_path)
    label_data = label_img.AsNumPy()


    print("- Computing Attenuation Map")
    if args.attenuation_function == "none":
        imp_map = np.ones(label_data.shape, dtype=np.float64)
        imp_map[label_data == 0] = 0
    else:
        if args.attenuation_function == "linear":
            imp_map = ift.ComputeLinearAttenuationWeightsByEDT(label_img,
                                                               args.max_attenuation_factor).AsNumPy()
        elif args.attenuation_function == "exp":
            imp_map = ift.ComputeExponentialAttenuationWeightsByEDT(label_img,
                                                                    args.max_attenuation_factor,
                                                                    args.exponent).AsNumPy()

    if args.invert_map:
        print("- Inverting Map")
        imp_map[label_data != 0] = 1.0 - imp_map[label_data != 0]
    
    # imp_map_int = imp_map * 255
    # ift.CreateImageFromNumPy(imp_map_int.astype(np.int32), True).Write("tmp/edt.nii.gz")

    imp_map = imp_map.astype(np.float64)
    imp_map = imp_map / imp_map.sum()

    print("- Writing Importance Map")
    np.save(args.out_importance_map_path, imp_map)



if __name__ == '__main__':
    main()
