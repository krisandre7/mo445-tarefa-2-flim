import pyift as ift
import matplotlib.pyplot as plt
import argparse
import sys
import os


def show(img, dpi=125):
    fig, ax = plt.subplots(dpi=dpi)
    ax.imshow(img)
    plt.show()


def imageOverlay(original, label):
    A = ift.Circular(1.0)
    tmp = ift.CopyImage(original)
    cmap = ift.CreateColorTable(ift.MaximumValue(label) + 1)
    ift.DrawLabels(tmp, label, cmap, A, True, 0.2)
    show(tmp.ToPlot())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment image using ISF Mix Mean")
    parser.add_argument("-i", "--input-image", help="input image", required=True)
    parser.add_argument("-s", "--superpixel", help="superpixel image", required=True)
    parser.add_argument("-m", "--objmap", help="objmap", required=True)
    parser.add_argument("-o", "--output-image", help="output image", required=True)
    parser.add_argument("-t", "--threshold", help="threshold for voting", required=True, type=float)
    args = vars(parser.parse_args())

    img = ift.ReadImageByExt(args["input_image"])
    suppxl = ift.ReadImageByExt(args["superpixel"])
    objmap = ift.ReadImageByExt(args["objmap"])

    segm = ift.SuperPixelMajorityVote(suppxl, objmap, args["threshold"])
    ift.WriteImageByExt(segm, args["output_image"])

    imageOverlay(img, segm)
