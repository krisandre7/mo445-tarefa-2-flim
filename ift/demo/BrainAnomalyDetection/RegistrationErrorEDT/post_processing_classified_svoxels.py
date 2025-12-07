import argparse
import timeit

import pyift.pyift as ift


def build_argparse():
    prog_desc = \
'''
Post-processing classified svoxels by removing those who do not have a given
minimum volume or mean registration error.
'''
    parser = argparse.ArgumentParser(
        description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-s', '--classified-svoxels-path', type=str, required=True,
                        help='Label Image with the supervoxels classified as lesion.')
    parser.add_argument('-r', '--reg-error-mag-path', type=str, required=True,
                        help='Registration Error Magnitudes of an image.')
    parser.add_argument('-o', '--post-classified-svoxels-path', type=str, required=True,
                        help='Output path for the post-processed .')
    parser.add_argument('-v', '--min-svoxel-volume', type=int, required=False, default=20,
                        help='Minimum Volume a Supervoxel must have. Default: 20')
    parser.add_argument('-f', '--min-mean-reg-error-mag-on-svoxel', type=float, required=False, default=50.0,
                        help='Minimum value for the Mean Reg. Error that s Supervoxel must have. Default: 50.0')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Classified SVoxels Path: {args.classified_svoxels_path}')
    print(f'- Reg Error Magnitude Path: {args.reg_error_mag_path}')
    print(f'- Output Path: {args.post_classified_svoxels_path}')
    print('--------------------------------------------')
    print(f'- Min. Volume: {args.min_svoxel_volume}')
    print(f'- Min. Mean Reg. Error on SVoxel: {args.min_mean_reg_error_mag_on_svoxel}')
    print('--------------------------------------------\n')


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    t1 = timeit.default_timer()

    reg_error_mag = ift.ReadImageByExt(args.reg_error_mag_path)
    svoxels_img = ift.ReadImageByExt(args.classified_svoxels_path)

    print("- Filtering SVoxels")
    filtered_svoxels_img = ift.RemoveSVoxelsByVolAndMeanRegError(svoxels_img, reg_error_mag,
                                                                 args.min_svoxel_volume, args.min_mean_reg_error_mag_on_svoxel)
    filtered_svoxels_img.Write(args.post_classified_svoxels_path)


    print("\nDone...")
    print(f"Time: {timeit.default_timer() - t1} secs")


if __name__ == "__main__":
    main()

