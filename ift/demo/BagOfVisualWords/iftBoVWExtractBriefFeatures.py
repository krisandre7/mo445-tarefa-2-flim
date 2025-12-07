# author: Cesar Castelo
# date: Mar 12, 2019
# description: This program computes the BRIEF features from a given image and a set of keypoints.
# It takes as parameters the img_filename, the keypoints_filename, the patch_size and the descriptor_size (number of features to be extracted).
# The extracted features are saved in a Numpy array whose name is received in local_feats_filename

import sys
import numpy as np
from skimage.feature import BRIEF
from skimage.io import imread
import ast
import subprocess

# main function
if __name__ == "__main__":

    # get the size of the terminal window
    terminal_rows, terminal_cols = subprocess.check_output(['stty', 'size']).decode().split()
    terminal_cols = int(terminal_cols)

    # verify input parameters
    if(len(sys.argv) != 6):
        print("#"*terminal_cols)
        print("usage: iftBoVWExtractBriefFeatures.py <...>")
        print("[1] img_filename: Filename of the image to extract the features from")
        print("[2] keypoints_filename: File containing the keypoints as an array (.csv)")
        print("[3] patch_size: Patch size to extract the features")
        print("[4] descriptor_size: Number of local features to be extracted")
        print("[5] local_feats_filename: Filename for the output file containing the local feautures extracted")
        print("#"*terminal_cols)
        sys.exit(-1)

    # read the input parameters
    img_filename = sys.argv[1]
    keypoints_filename = sys.argv[2]
    patch_size = sys.argv[3]
    descriptor_size = sys.argv[4]
    local_feats_filename = sys.argv[5]

    # read image and keypoints
    img = imread(img_filename, as_gray=True)
    keypoints = np.genfromtxt(keypoints_filename, delimiter=",")

    # extract BRIEF features
    extractor = BRIEF(patch_size=int(patch_size), descriptor_size=int(descriptor_size))
    extractor.extract(img, keypoints)
    local_feats = extractor.descriptors

    # save local feat vectors
    np.savetxt(local_feats_filename, local_feats, delimiter=",")

