import argparse
import sys
import timeit

import numpy as np

import pyift.pyift as ift


def build_argparse():
    prog_desc = \
        '''
Extract supervoxels for a given registered image on its Registration Errors.
This script performs the ISF method for an input image by taking into account the image concatenated with the 
template as the input multi-band image for ISF.
The initial seeds are obtained by thresholding the registration error magnitude and getting the maxima of
each resulting component.
A label mask is required so that the svoxel segmentation is performed inside each object independently and 
their results are combined into a single label image at the end.
The parameters used by ISF can be passed for each object of the mask or a general set of them for all objects of the mask.
Small Supervoxels near from the object' surface whose volume is less than a given threshold (optionally passed) can be removed.
'''
    parser = argparse.ArgumentParser(
        description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--image', type=str,
                        required=True, help='Registered Image pathname..')
    parser.add_argument('-r', '--reg-error-mag', type=str, required=True,
                        help='Registration Error Magnitudes of the input image.')
    parser.add_argument('-t', '--template', type=str,
                        required=True, help='Template (Reference Image).')
    parser.add_argument('-m', '--label-image', type=str, required=True,
                        help='Label Image with with the target object where the svoxel segmentation will happen.')
    parser.add_argument('-o', '--out-supervoxel-image', type=str,
                        required=True, help='Output Supervoxel Image.')

    parser.add_argument('-a', '--alpha-factors', type=float, nargs="+", default=[0.08],
                        help='Alpha factor of ISF, one for each Object of the Label Image. '
                             'Ex: -a 0.08,0.1,0.09 (3 objects) or -a 0.8 (same value for all objects). Default: 0.08')
    parser.add_argument('-b', '--beta-factors', type=float, nargs="+", default=[5],
                        help='Beta factor of ISF, one for each Object of the Label Image. '
                             'Ex: -b 5,2,4 (3 objects) or -b 5 (same value for all objects). Default: 5')
    parser.add_argument('-f', '--threshold-otsu-factors', type=float, nargs="+", default=[0.5],
                        help='Factor used to increase/decrease on otsu threshold to binarized reg error components. '
                             'Ex: -f 0.5,1.0,2.0 (3 objects) or -f 0.5 (same value for all objects). Default: 0.5')
    parser.add_argument('-d', '--min-euclidean-distances', type=float, nargs="+", default=[5],
                        help='Minimum euclidean distance to the each target object borders that the initial ISF seeds '
                             'must have. Ex: -d 0,1,2 (3 objects) or -f 5 (same value for all objects). '
                             'Default: 0.0 (no restriction).')
    parser.add_argument('-y', '--number-seeds-on-correct-regions', type=int, nargs="+", default=[50],
                        help='Number of Seeds on Correct Regions. Ex: -y 50,50,100 (3 objects) or -y 50 '
                             '(same value for all objects). Default: 50')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Image: {args.image}')
    print(f'- Registration Error Magnitude: {args.reg_error_mag}')
    print(f'- Template: {args.template}')
    print(f'- Label Image: {args.label_image}')
    print(f'- Output Supervoxel Image: {args.out_supervoxel_image}')
    print('--------------------------------------------')
    print(f'- Alpha Factors: {args.alpha_factors}')
    print(f'- Beta Factors: {args.beta_factors}')
    print(f'- Threshold Otsu Factors: {args.threshold_otsu_factors}')
    print(
        f'- Min. Euclidean Distance from initial seeds to Object Borders: {args.min_euclidean_distances}')
    print(
        f'- Number of Seeds on Correct Regions: {args.number_seeds_on_correct_regions}')
    print('--------------------------------------------\n')


def validate_args(args, label_img):
    data = label_img.AsNumPy()
    objs = np.unique(data[data != 0])

    if len(args.alpha_factors) != len(objs):
        if len(args.alpha_factors) == 1:
            args.alpha_factors = args.alpha_factors * len(objs)
        else:
            sys.exit(f"Error: Number of Alpha Factors {len(args.alpha_factors)} is != the number of Objects in the "
                     f"mask: {len(objs)}")

    if len(args.beta_factors) != len(objs):
        if len(args.beta_factors) == 1:
            args.beta_factors = args.beta_factors * len(objs)
        else:
            sys.exit(f"Error: Number of Beta Factors {len(args.beta_factors)} is != the number of Objects in the "
                     f"mask: {len(objs)}")

    if len(args.threshold_otsu_factors) != len(objs):
        if len(args.threshold_otsu_factors) == 1:
            args.threshold_otsu_factors = args.threshold_otsu_factors * \
                len(objs)
        else:
            sys.exit(f"Error: Number of Otsu Thresholds {len(args.threshold_otsu_factors)} is != the number "
                     f"of Objects in the mask: {len(objs)}")

    if len(args.min_euclidean_distances) != len(objs):
        if len(args.min_euclidean_distances) == 1:
            args.min_euclidean_distances = args.min_euclidean_distances * \
                len(objs)
        else:
            sys.exit(f"Error: Number of Min. Euclidean Distance {len(args.min_euclidean_distances)} is != the number "
                     f"of Objects in the mask: {len(objs)}")

    if len(args.number_seeds_on_correct_regions) != len(objs):
        if len(args.number_seeds_on_correct_regions) == 1:
            args.number_seeds_on_correct_regions = args.number_seeds_on_correct_regions * \
                len(objs)
        else:
            sys.exit(f"Error: Number of Seeds on Correct Regions {len(args.number_seeds_on_correct_regions)} is != "
                     f"the number of Objects in the mask: {len(objs)}")


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    label_img = ift.ReadImageByExt(args.label_image)
    validate_args(args, label_img)
    print_args(args)

    t1 = timeit.default_timer()

    img = ift.ReadImageByExt(args.image)
    reg_error_mag = ift.ReadImageByExt(args.reg_error_mag)
    template = ift.ReadImageByExt(args.template)
    label_img = ift.ReadImageByExt(args.label_image)

    # inserting an arbitrary value at the beginning of the parameter arrays in order to make
    # each index [i] has the parameter for the object i
    alpha_factors = np.insert(args.alpha_factors, 0, -1)
    beta_factors = np.insert(args.beta_factors, 0, -1)
    threshold_otsu_factors = np.insert(args.threshold_otsu_factors, 0, -1)
    min_euclidean_distances = np.insert(args.min_euclidean_distances, 0, -1)
    number_seeds_on_correct_regions = np.insert(
        args.number_seeds_on_correct_regions, 0, -1).astype(np.int32)

    svoxel_img = ift.ISFOnRegErrors(img, reg_error_mag, template, label_img,
                                    ift.CreateDblArrayFromNumPy(alpha_factors),
                                    ift.CreateDblArrayFromNumPy(beta_factors),
                                    ift.CreateDblArrayFromNumPy(
                                        threshold_otsu_factors),
                                    ift.CreateDblArrayFromNumPy(
                                        min_euclidean_distances),
                                    ift.CreateIntArrayFromNumPy(number_seeds_on_correct_regions))

    t2 = timeit.default_timer()

    svoxel_img.Write(args.out_supervoxel_image)

    print("\nDone...")
    print(f"Time: {t2 - t1} secs")


if __name__ == "__main__":
    main()
