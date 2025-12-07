import argparse
import timeit

import pyift.pyift as ift


def build_argparse():
    prog_desc = \
'''
Extracts the LBP feats for each supervoxel from a given image by using the algorithm VLBP.
It creates a specific dataset for each supervoxel so that each dataset has one sample extracted for the
test image and one sample per training image from a training control set.
"The filename of each dataset is supervoxel_XXXX.zip, where XXXX is the supervoxel label, starting at 0001.
'''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--image', type=str, required=True,
                        help='Image.')
    parser.add_argument('-s', '--supervoxels', type=str, required=True,
                        help='Label Image with the supervoxels for the image.')
    parser.add_argument('-x', '--training-set', type=str, required=True,
                        help='Directory or CSV with the pathnames from the training image set.')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='Output Directory where the supervoxel datasets will be saved.')
    parser.add_argument('-n', '--bins', type=int, default=128,
                        help='Number of bins of the VLBP. Default: 128.')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Image: {args.image}')
    print(f'- Supervoxels Path: {args.supervoxels}')
    print(f'- Training Set: {args.training_set}')
    print(f'- Output Dir: {args.output_dir}')
    print('--------------------------------------------')
    print(f'- Number of Bins: {args.bins}')
    print('--------------------------------------------\n')



def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    t1 = timeit.default_timer()

    image = ift.ReadImageByExt(args.image)
    svoxels_img = ift.ReadImageByExt(args.supervoxels)
    train_set = ift.LoadFileSetFromDirOrCSV(args.training_set, 0, True)

    n_supervoxels = ift.MaximumValue(svoxels_img)

    print("- Building Datasets")
    Zarr = ift.ExtractSupervoxelVLBPFeats(image, svoxels_img, train_set, args.bins, None)    

    print("- Writing Supervoxel Datasets")
    ift.WriteSupervoxelDataSets(Zarr, n_supervoxels, args.output_dir)

    print("\nDone...")
    print(f"Time: {timeit.default_timer() - t1} secs")


if __name__ == "__main__":
    main()
