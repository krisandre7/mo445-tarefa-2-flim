import pyift as ift
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Connect object markers through optimum path")
    parser.add_argument("-i", "--input-image", help="input/output dataset", required=True)
    parser.add_argument("-m", "--input-markers", help="input markers", required=True)
    parser.add_argument("-o", "--output-markers", help="output markers", required=True)
    args = vars(parser.parse_args())

    img = ift.ReadImageByExt(args["input_image"])
    seeds = ift.ReadSeeds(img, args["input_markers"])
    out_seeds = ift.ConnectObjectSeeds(img, seeds)
    ift.WriteSeeds(out_seeds, img,args["output_markers"])
