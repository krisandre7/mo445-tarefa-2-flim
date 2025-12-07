import argparse
import timeit

import pyift.pyift as ift


def build_argparse():
    prog_desc = \
        '''
        Extracts the histogram of registration error magnitudes for each supervoxel of a given test image.
        It creates a specific dataset for each supervoxel so that each dataset has one sample extracted for the
        test image and one sample per training image from a training control set.
        A bias can be passed to attenuate the registration error magnitude in the test image.
        "The filename of each dataset is supervoxel_XXXX.zip, where XXXX is the supervoxel label, starting at 0001.
        '''
    parser = argparse.ArgumentParser(
        description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--test-image', type=str, required=True,
                        help='Test Image used to build the supervoxel datasets.')
    parser.add_argument('-s', '--test-supervoxels', type=str, required=True,
                        help='Label Image with the supervoxels for the test image.')
    parser.add_argument('-t', '--template', type=str,
                        required=True, help='Template (Reference Image).')
    parser.add_argument('-x', '--training-set', type=str, required=True,
                        help='Directory or CSV with the pathnames from the training control images.')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='Output Directory where the symmetry supervoxel datasets will be saved.')
    parser.add_argument('-n', '--bins', type=int, default=128,
                        help='Number of bins of the histogram. Default: 128.')
    parser.add_argument('-b', '--bias', type=str, help='Pathname from the bias (e.g. normal reg error magnitude) used '
                                                       'to attenuate the resulting registration error magnitude')
    parser.add_argument('-d', '--dilation-radius', type=float, default=0.0,
                        help='Dilation Radius to dilate the svoxels before extracting features. Default: 0.0 (no dilation)')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Test Image Path: {args.test_image}')
    print(f'- Test Supervoxel Path: {args.test_supervoxels}')
    print(f'- Template: {args.template}')
    print(f'- Training Set: {args.training_set}')
    print(f'- Output Dir: {args.output_dir}')
    print('--------------------------------------------')
    print(f'- Number of Bins: {args.bins}')
    if args.bias:
        print(f'- Bias: {args.bias}')
    print(f"- Dilation Radius: {args.dilation_radius}")
    print('--------------------------------------------\n')


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    t1 = timeit.default_timer()

    test_img = ift.ReadImageByExt(args.test_image)
    test_svoxels_img = ift.ReadImageByExt(args.test_supervoxels)
    train_set = ift.LoadFileSetFromDirOrCSV(args.training_set, 0, True)
    template = ift.ReadImageByExt(args.template)
    bias = ift.ReadImageByExt(args.bias) if args.bias else None

    n_supervoxels = ift.MaximumValue(test_svoxels_img)

    print("- Building Datasets")
    if args.dilation_radius <= 0.0:
        print("\t- No dilation")
        Zarr = ift.ExtractSupervoxelHistRegErrorsFeats(test_img, test_svoxels_img, train_set,
                                                       template, args.bins, bias, None)
    else:
        print("\t- Dilation")
        Zarr = ift.ExtractSupervoxelHistRegErrorsFeatsDilation(test_img, test_svoxels_img, train_set,
                                                               template, args.bins, args.dilation_radius,
                                                               bias, None)

    print("- Building Reference Data")
    ift.BuildRefDataSupervoxelDataSets(
        Zarr, n_supervoxels, args.test_image, args.test_supervoxels, train_set)

    print("- Writing Supervoxel Datasets")
    ift.WriteSupervoxelDataSets(Zarr, n_supervoxels, args.output_dir)

    print("\nDone...")
    print(f"Time: {timeit.default_timer() - t1} secs")


if __name__ == "__main__":
    main()
