import pyift.pyift as ift
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Label dataset samples according to marker labels")
    parser.add_argument("-d", "--dataset", help="input/output dataset", required=True)
    parser.add_argument("-s", "--superpixel-image", help="input superpixel image", required=True)
    parser.add_argument("-m", "--markers", help="input markers", required=True)
    args = vars(parser.parse_args())

    Z = ift.ReadDataSet(args["dataset"])
    suppxl = ift.ReadImageByExt(args["superpixel_image"])
    seeds = ift.ReadSeeds(suppxl, args["markers"])
    ift.LabelDataSetFromSeeds(Z, seeds, suppxl)
    ift.WriteDataSet(Z, args["dataset"])
