import argparse
import timeit

import pyift.pyift as ift


def build_argparse():
    prog_desc = \
        '''
        Computes the Magnitude of the Registration Error --- voxel-wise absolute difference -- between
        a registered image and its template.
        A Bias image (e.g. the normal reg. error magnitude) can be passed to attenuate the registration error 
        Magnitude.
        If a mask is passed, the resulting registration error mag. will be only considered inside it.
        '''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--image', type=str, required=True, help='Pathname from the registered Image.')
    parser.add_argument('-t', '--template', type=str, required=True, help='Template (Reference Image).')
    parser.add_argument('-o', '--out-reg-error-mag', type=str, required=True,
                        help='Pathname for the resulting registration error magnitude.')
    parser.add_argument('-m', '--mask', type=str, help='Pathname with the mask of the target objects.')
    parser.add_argument('-b', '--bias', type=str, help='Pathname from the bias (e.g. normal reg error magnitude) used '
                                                       'to attenuate the resulting registration error magnitude')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Image Path: {args.image}')
    print(f'- Template: {args.template}')
    print(f'- Output Registration Error Magnitude: {args.out_reg_error_mag}')
    print('--------------------------------------------')
    if args.mask:
        print(f'- Mask: {args.mask}')
    if args.bias:
        print(f'- Bias: {args.bias}')
    print('--------------------------------------------\n')



def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    t1 = timeit.default_timer()

    img = ift.ReadImageByExt(args.image)
    template = ift.ReadImageByExt(args.template)
    bias = ift.ReadImageByExt(args.bias) if args.bias else None
    mask = ift.ReadImageByExt(args.mask) if args.mask else None

    reg_error_mag = ift.RegErrorMagnitude(img, template, bias)

    if mask:
        reg_error_mag = ift.Mask(reg_error_mag, mask)

    reg_error_mag.Write(args.out_reg_error_mag)

    print("\nDone...")
    print(f"Time: {timeit.default_timer() - t1} secs")


if __name__ == "__main__":
    main()
