#!/usr/bin/python

# author: Cesar Castelo
# date: Nov 26, 2019
# description: This program takes a training set and performs part of the BoVW pipeline (interest point detection and local feats extraction)
# The program iftBoVWLearn is run on the training set to and then the local feats are saved in a IFT dataset

import os, sys
import shutil
import subprocess
from math import sqrt
import json
import numpy as np
import zipfile
import tempfile
from iftBoVWCommonFunctions import create_func_params_json, convert_method_names_to_numbers

patch_size = 9
stride = 7
col_space = 'lab'

# main function
if __name__ == "__main__":

    # get the size of the terminal window
    terminal_rows, terminal_cols = subprocess.check_output(['stty', 'size']).decode().split()
    terminal_cols = int(terminal_cols)

    # verify input parameters
    if(len(sys.argv) != 10):
        print("#"*terminal_cols)
        print("usage: iftBoVWLocalFeatsExtraction.py <...>")
        print("[1] dataset_dirname: Dirname of the image dataset")
        print("[2] image_set_name: Image set to be used among the ones in the dataset ['learn_001','train_001','test_001', etc]")
        print("[3] use_samp_masks: Use masks for sampling ['yes', 'no']")
        print("[4] int_point_detection_method: Method to detect interest points")
        print("    'random': Random sampling")
        print("    'grid': Grid sampling")
        print("    'unsup-spix-isf': Unsupervised superpixels by ISF")
        print("    'sup-spix-isf': Supervised superpixels by ISF")
        print("[5] feat_extraction_method: Method to extract the local features")
        print("    'raw': Raw pixel values")
        print("    'bic': Border/Interior Pixel Classification (BIC)")
        print("    'lbp': Local Binary Patterns (LBP)")
        print("    'brief': BRIEF descriptor")
        print("    'conv': Convolutional features")
        print("    'conv-ml': Multi Layer Convolutional features")
        print("    'deep-feats-mimg': Deep features from mimage")
        print("[6] save_patches: Save the patches that will be extracted from the images [0: No, 1: Yes]")
        print("[7] perform_tsne: Perform t-SNE projection [0-No, 1-Yes]")
        print("[8] save_extra_info: Save extra info from the local features [0: No, 1: Yes]")
        print("[9] output_suffix: Suffix to be added to the output files")
        print("#"*terminal_cols)
        sys.exit(-1)

    # read input parameters
    dataset_dirname = sys.argv[1].rstrip("/")
    image_set_name = sys.argv[2]
    use_samp_masks = sys.argv[3]
    int_point_detection_method = sys.argv[4]
    feat_extraction_method = sys.argv[5]
    save_patches = sys.argv[6]
    perform_tsne = sys.argv[7]
    save_extra_info = sys.argv[8]
    output_suffix = sys.argv[9]
    
    # print input parameters
    print("#"*terminal_cols)
    print("Input parameters (iftBoVWLocalFeatsExtraction.py script):")
    print("#"*terminal_cols)
    print("dataset_dirname:", dataset_dirname)
    print("image_set_name:", image_set_name)
    print("use_samp_masks:", use_samp_masks)
    print("int_point_detection_method:", int_point_detection_method)
    print("feat_extraction_method:", feat_extraction_method)
    print("save_patches:", save_patches)
    print("perform_tsne:", perform_tsne)
    print("save_extra_info:", save_extra_info)
    print("output_suffix:", output_suffix)

    output_dir_basename = "local_feats_" + dataset_dirname
    output_dirname = output_dir_basename + '_' + int_point_detection_method + '_' + feat_extraction_method
    output_suffix = image_set_name + "_" + output_suffix

    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)

    # create JSON with function params
    func_params = create_func_params_json(int_point_detection_method, feat_extraction_method, "", "", "",
        use_samp_masks, patch_size, stride, col_space)

    # convert method names to numbers (according to   definition in LibIFT)
    [int_point_detection_method, feat_extraction_method, _, _, _] = convert_method_names_to_numbers(int_point_detection_method,
        feat_extraction_method, "", "", "")

    # set filenames for input files
    image_set_filename = os.path.join(dataset_dirname, image_set_name + ".csv")
    image_set_mask_filename = "-1"
    if(use_samp_masks == "yes"):
        image_set_mask_filename = os.path.join(dataset_dirname, image_set_name + "_masks.csv")
        
    # execute iftBoVWLocalFeatsExtraction
    print("RUNNING iftBoVWLocalFeatsExtraction ...")
    print("#"*terminal_cols)
    cmd = "iftBoVWLocalFeatsExtraction {} {} {} {} '{}' {} {} {} {} {}".format(image_set_filename, image_set_mask_filename, int_point_detection_method,
        feat_extraction_method, json.dumps(func_params, sort_keys=False), save_patches, perform_tsne, save_extra_info, output_dir_basename, output_suffix)
    if (os.system(cmd) != 0):
            sys.exit("Error in iftBoVWLocalFeatsExtraction")