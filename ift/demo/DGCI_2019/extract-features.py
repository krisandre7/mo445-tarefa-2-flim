import pyift.pyift as ift
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract image features for each superpixel")
    parser.add_argument("-i", "--input-image", help="input image", required=True)
    parser.add_argument("-s", "--superpixel-image", help="input superpixel image", required=True)
    parser.add_argument("-c", "--extract-coord", help="extract superpixel coordinates", type=str2bool, required=False, default=False)
    parser.add_argument("-d", "--dataset", help="output dataset file", required=True)
    args = vars(parser.parse_args())

    img = ift.ReadImageByExt(args["input_image"])
    suppxl = ift.ReadImageByExt(args["superpixel_image"])
    A = ift.Circular(5.0)
    feat = ift.ExtractImageFeatures(img, suppxl, A, args["extract_coord"])
    Z = ift.MImageToDataSet(feat, suppxl)
    ift.SetStatus(Z, 0x04);
    ift.WriteDataSet(Z, args["dataset"])
    
