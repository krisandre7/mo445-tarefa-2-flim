import argparse
import timeit

import pyift.pyift as ift


def build_argparse():
    prog_desc = \
        '''
        Computes the mean Registration Error Magnitude --- voxel-wise absolute difference -- for a set of " 
        "registered images and the used template.
        Optionally, the standard deviation reg. error magnitude can be added to the resulting mean.
        If a mask is passed, the output registration error mag. will be only considered inside it.
        '''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--image-set', type=str, required=True,
                        help='Directory of CSV with the registered image pathnames.')
    parser.add_argument('-t', '--template', type=str, required=True, help='Template (Reference Image).')
    parser.add_argument('-o', '--out-mean-reg-error-mag', type=str, required=True,
                        help='Pathname for the resulting mean registration error magnitude.')
    parser.add_argument('-m', '--mask', type=str, help='Pathname with the mask of the target objects.')
    parser.add_argument('--use-stdev', action='store_true',
                        help='Add/Use the Standard Deviation Registration Error Magnitudes into the output map.')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Image Set: {args.image_set}')
    print(f'- Template: {args.template}')
    print(f'- Output Mean Registration Error Magnitude: {args.out_mean_reg_error_mag}')
    print('--------------------------------------------')
    if args.mask:
        print(f'- Mask: {args.mask}')
    print(f'- Add/Use Standard Deviation: {args.use_stdev}')
    print('--------------------------------------------\n')



def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    t1 = timeit.default_timer()

    img_set = ift.LoadFileSetFromDirOrCSV(args.image_set, 0, True)
    template = ift.ReadImageByExt(args.template)
    mask = ift.ReadImageByExt(args.mask) if args.mask else None

    mean_reg_error_mag = ift.MeanRegErrorMagnitude(img_set, template, args.use_stdev)

    if mask:
        mean_reg_error_mag = ift.Mask(mean_reg_error_mag, mask)

    mean_reg_error_mag.Write(args.out_mean_reg_error_mag)

    print("\nDone...")
    print(f"Time: {timeit.default_timer() - t1} secs")


if __name__ == "__main__":
    main()
