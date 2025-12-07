import argparse
import timeit

import pyift.pyift as ift


def build_argparse():
    prog_desc = \
        '''
Relabel a given supervoxel map.
        '''
    parser = argparse.ArgumentParser(
        description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--supervoxels', type=str, required=True,
                        help='Label Image with the supervoxels.')
    parser.add_argument('-o', '--out-supervoxels', type=str, required=True,
                        help='Output Label Image with the relabeled supervoxels.')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Supervoxel Map: {args.supervoxels}')
    print(f'- Output Supervoxels: {args.out_supervoxels}')
    print('--------------------------------------------\n')


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    print("- Reading SVoxels")
    svoxels_img = ift.ReadImageByExt(args.supervoxels)

    print("- Relabeling SVoxels")
    svoxels_relabeled = ift.FastLabelComp(svoxels_img, ift.Spheric(1.74))

    svoxels_relabeled.Write(args.out_supervoxels)


if __name__ == "__main__":
    main()
