import pyift as ift
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a labeled set of markers according to a dataset")
    parser.add_argument("-d", "--input-dataset", help="input dataset", required=True)
    parser.add_argument("-s", "--superpixel-image", help="input superpixel image", required=True)
    parser.add_argument("-o", "--output-markers", help="output markers", required=True)
    args = vars(parser.parse_args())

    Z = ift.ReadDataSet(args['input_dataset'])
    suppxl = ift.ReadImageByExt(args['superpixel_image'])
    seeds = ift.DataSetToLabeledSeeds(Z, suppxl)
    ift.WriteSeeds(args['output_markers'], seeds, suppxl)