import argparse
import timeit

import pyift.pyift as ift


def build_argparse():
    prog_desc = \
'''
Extracts the histogram of registration error magnitudes for each supervoxel of a given test image.
It creates a specific dataset for each supervoxel so that each dataset has one sample extracted for the
test image and one sample per training image from a training control set.
"The filename of each dataset is supervoxel_XXXX.zip, where XXXX is the supervoxel label, starting at 0001.
'''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-r', '--test-reg-error-mag', type=str, required=True,
                        help='Registration Error Magnitudes of an image.')
    parser.add_argument('-s', '--test-supervoxels', type=str, required=True,
                        help='Label Image with the supervoxels for the test image.')
    parser.add_argument('-x', '--training-reg-error-mag-set', type=str, required=True,
                        help='Directory or CSV with the pathnames from the training control registration error magnitudes.')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='Output Directory where the supervoxel datasets will be saved.')
    parser.add_argument('-n', '--bins', type=int, default=128,
                        help='Number of bins of the histogram. Default: 128.')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Test Registration Error Magnitude Path: {args.test_reg_error_mag}')
    print(f'- Test Supervoxel Path: {args.test_supervoxels}')
    print(f'- Training Reg. Error Mag. Set: {args.training_reg_error_mag_set}')
    print(f'- Output Dir: {args.output_dir}')
    print('--------------------------------------------')
    print(f'- Number of Bins: {args.bins}')
    print('--------------------------------------------\n')



def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    t1 = timeit.default_timer()

    test_reg_error_mag = ift.ReadImageByExt(args.test_reg_error_mag)
    test_svoxels_img = ift.ReadImageByExt(args.test_supervoxels)
    train_set = ift.LoadFileSetFromDirOrCSV(args.training_reg_error_mag_set, 0, True)

    n_supervoxels = ift.MaximumValue(test_svoxels_img)

    print("- Building Datasets")
    Zarr = ift.ExtractSupervoxelHistRegErrorsFeats(test_reg_error_mag, test_svoxels_img, train_set, args.bins, None)    

    print("- Writing Supervoxel Datasets")
    ift.WriteSupervoxelDataSets(Zarr, n_supervoxels, args.output_dir)

    print("\nDone...")
    print(f"Time: {timeit.default_timer() - t1} secs")


if __name__ == "__main__":
    main()
