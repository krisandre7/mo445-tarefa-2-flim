import argparse
import math
import os
import timeit

import numpy as np
from keras.models import load_model
from mykerasmodels import predict

import pyift.pyift as ift



def build_argparse():
    prog_desc = \
'''
Convolve an image set with a kernel bank (in a keras model).
'''

    parser = argparse.ArgumentParser(description=prog_desc,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--image-set', type=str, required=True,
                        help='Image set to convolve.')
    parser.add_argument('-k', '--keras-model', type=str, required=True,
                        help='Keras model (*.h5) with the kernel bank to '
                             'convolve the image set.')
    parser.add_argument('-o', '--out-dir', type=str, required=True,
                        help='Output directory where the multi-band filtered '
                              'images will be saved for the input image set')
    parser.add_argument('-m', '--mask', type=str,
                        help='Mask for the input image whose the min. '
                             'bounding box in the objects defines the input '
                             'shape for the convolutions in the keras model')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Image Set Entry: {args.image_set}')
    print(f'- Model: {args.keras_model}')
    print('--------------------------------------------')
    if args.mask:
        print(f'- Mask: {args.mask}')
    print('--------------------------------------------\n')


def build_batch_data(batch_idx, batch_size, img_paths, input_shape, mbb=None):
    start = batch_idx * batch_size
    end = (batch_idx + 1) * batch_size

    data = []

    for path in img_paths[start : end]:
        print(path)
        img = ift.ReadImageByExt(path)

        if mbb:
            img_data = ift.ExtractROI(img, mbb).AsNumPy()
        else:
            img_data = img.AsNumPy()

        img_data = np.reshape(img_data, (*img_data.shape, 1))

        if img_data.shape != input_shape:
            print("Image with Different Shape of the Model's Input Layer: " \
                  f"{img_data.shape} != {input_shape}... Ignoring it")
            continue
        
        data.append(img_data)

    return np.array(data)


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    print('- Loading Keras 3D Convolutional Model')
    model = load_model(args.keras_model)
    # (zsize, ysize, xsize, n_channels)
    input_shape = model.layers[0].input_shape[1:]

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    mbb = None
    if args.mask:
        mask = ift.ReadImageByExt(args.mask)
        mbb = ift.MinBoundingBox(mask, None)

    img_paths = ift.LoadFileSetFromDirOrCSV(args.image_set, 0, True).GetPaths()

    t1 = timeit.default_timer()

    for path in img_paths:
        print(path)
        img = ift.ReadImageByExt(path)

        if mbb:
            img_data = ift.ExtractROI(img, mbb).AsNumPy()
        else:
            img_data = img.AsNumPy()
        
        filt_data = predict(img_data, model)
        print(filt_data.shape)

        img_filename = os.path.basename(path)
        array_filename = img_filename.replace(".nii.gz", ".npy")
        out_path = os.path.join(args.out_dir, array_filename)
        np.save(out_path, filt_data)

    print(f"\ntime = {timeit.default_timer() - t1} secs")


if __name__ == "__main__":
    main()
