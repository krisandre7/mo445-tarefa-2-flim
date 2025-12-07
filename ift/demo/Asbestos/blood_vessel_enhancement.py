#!/usr/bin/env python

import argparse

import itk
from distutils.version import StrictVersion as VS
import pyift.pyift as ift
import numpy as np

if VS(itk.Version.GetITKVersion()) < VS("5.0.0"):
    print("ITK 5.0.0 or newer is required.")
    sys.exit(1)

parser = argparse.ArgumentParser(description="Segment blood vessels.")
parser.add_argument("input_image")
parser.add_argument("output_image")
parser.add_argument("--sigma", type=float, default=1.0)
parser.add_argument("--alpha1", type=float, default=0.5)
parser.add_argument("--alpha2", type=float, default=2.0)
args = parser.parse_args()

ift_image = ift.ReadImageByExt(args.input_image)
np_image = ift_image.AsNumPy().astype(np.float64)
#print(np_image)
#print(np_image.dtype)
input_image = itk.GetImageFromArray(np_image)
#print(input_image)

#input_image = itk.imread(args.input_image, itk.ctype("float"))
#print(input_image)

hessian_image = itk.hessian_recursive_gaussian_image_filter(
    input_image, sigma=args.sigma
)
print(hessian_image)

vesselness_filter = itk.Hessian3DToVesselnessMeasureImageFilter[
    itk.ctype("float")
].New()
vesselness_filter.SetInput(hessian_image)
vesselness_filter.SetAlpha1(args.alpha1)
vesselness_filter.SetAlpha2(args.alpha2)

itk.imwrite(vesselness_filter, args.output_image)
