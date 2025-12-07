import argparse
import timeit

import pyift.pyift as ift


def build_argparse():
    prog_desc = \
'''
Extracts the texture features for each supervoxel from a given test image and its attention map.
Texture features corresponds to the concatenation of:
histogram of values in the attention map (n_bins feats), and VLBP feats of image (n_bins feats)

It creates a specific dataset for each supervoxel so that each dataset has one sample extracted for the
test image and one sample per training image from a training control set.
"The filename of each dataset is supervoxel_XXXX.zip, where XXXX is the supervoxel label, starting at 0001.
'''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--image', type=str, required=True,
                        help='Input image.')
    parser.add_argument('-a', '--attention-map', type=str, required=True,
                        help='Attention Map of the input image image.')
    parser.add_argument('-s', '--supervoxels', type=str, required=True,
                        help='Label Image with the supervoxels for the input image.')
    parser.add_argument('-t', '--training-set', type=str, required=True,
                        help='Directory or CSV with the pathnames from the training control images.')
    parser.add_argument('-x', '--training-attention-map-set', type=str, required=True,
                        help='Directory or CSV with the pathnames from the training control attention maps.')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='Output Directory where the supervoxel datasets will be saved.')
    parser.add_argument('-n', '--bins', type=int, default=128,
                        help='Number of bins of the histogram. Default: 128.')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Image Path: {args.image}')
    print(f'- Attention Map Path: {args.attention_map}')
    print(f'- Supervoxels Path: {args.supervoxels}')
    print(f'- Training Set: {args.training_set}')
    print(f'- Training Attention Map Set: {args.training_attention_map_set}')
    print(f'- Output Dir: {args.output_dir}')
    print('--------------------------------------------')
    print(f'- Number of Bins: {args.bins}')
    print('--------------------------------------------\n')



def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    t1 = timeit.default_timer()

    img = ift.ReadImageByExt(args.image)
    attention_map = ift.ReadImageByExt(args.attention_map)
    svoxels_img = ift.ReadImageByExt(args.supervoxels)
    training_set = ift.LoadFileSetFromDirOrCSV(args.training_set, 0, True)
    training_attention_map_set = ift.LoadFileSetFromDirOrCSV(args.training_attention_map_set, 0, True)

    n_supervoxels = ift.MaximumValue(svoxels_img)

    print("- Building Datasets")
    Zarr = ift.ExtractSupervoxelTextureFeats(img, attention_map, svoxels_img,
                                             training_set, training_attention_map_set,
                                             args.bins, None)

    print("- Writing Supervoxel Datasets")
    ift.WriteSupervoxelDataSets(Zarr, n_supervoxels, args.output_dir)

    print("\nDone...")
    print(f"Time: {timeit.default_timer() - t1} secs")


if __name__ == "__main__":
    main()
