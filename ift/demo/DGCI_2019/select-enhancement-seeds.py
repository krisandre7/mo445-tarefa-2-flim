import pyift.pyift as ift
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select better seeds for object enhancement given a cluster label image")
    parser.add_argument("-s", "--input-markers", help="input seeds", dest="input", required=True)
    parser.add_argument("-l", "--cluster-label-image", help="cluster label image", dest="comp", required=True)
    parser.add_argument("-o", "--output-markers", help="output markers", dest="output", required=True)
    parser.add_argument("-t", "--threshold", help="label mixture threshold", type=float, default=0.5, required=False)
    args = vars(parser.parse_args())

    comp = ift.ReadImageByExt(args['comp'])
    seeds = ift.ReadSeeds(comp,args['input'])
    out = ift.SelectSeedsForEnhancement(seeds, comp, 1, args['threshold'])
    ift.WriteSeeds(out,comp,args['output'])
