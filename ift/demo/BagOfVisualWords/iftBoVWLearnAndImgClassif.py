#!/usr/bin/python

# author: Cesar Castelo
# date: Mar 03, 2019
# description: This program takes a number of training sets and testing sets and perform the BoVW pipeline (including SVM classification)
# First, the program iftBoVWLearn is run on each training set to learn a BoVW descritor. Then, using each BoVW descriptor that was learnt
# using each learning set, the program iftBoVWImgClassif is run to compute the feature vectors of each training set and each testing set

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
        print("usage: iftBoVWLearnAndImgClassif.py <...>")
        print("[1] dataset_dirname: Dirname of the image dataset for learning, training and testing tasks")
        print("[2] n_splits: Number of splits in the dataset")
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
        print("[6] dict_estimation_method: Method to estimate the dictionary")
        print("    'unsup-kmeans': Unsup kMeans")
        print("    'sup-kmeans-pix-label': Sup k-means with ordering by pixel label")
        print("    'sup-kmeans-img-class': Sup k-means with ordering by image class")
        print("    'sup-kmeans-image': Sup k-means with ordering by image")
        print("    'sup-kmeans-position': Sup k-means with ordering by position")
        print("    'sup-kmeans-img-class-position': Sup k-means with ordering by image class and position")
        print("    'unsup-opf': Unsup OPF")
        print("    'sup-opf-pix-label': Sup OPF with ordering by pixel label")
        print("    'sup-opf-img-class': Sup OPF with ordering by image class")
        print("    'sup-opf-image': Sup OPF with ordering by image")
        print("    'sup-opf-position': Sup OPF with ordering by position")
        print("    'sup-opf-img-class-position': Sup OPF with ordering by image class and position")
        print("    'sup-manual-img-class': Sup Manual with ordering by image class")
        print("    'sup-manual-position': Sup Manual with ordering by position")
        print("    'sup-manual-img-class-position': Sup Manual with ordering by image class and position")
        print("[7] coding_method: Method to perform coding operation")
        print("    'hard-asgmt': Hard assignment clustering")
        print("    'soft-asgmt': Soft assignment clustering")
        print("    'hard-asgmt-batch': Hard assignment clustering with batch processing")
        print("    'soft-asgmt-batch': Soft assignment clustering with batch processing")
        print("    '2nd-order-stats-fv': 2nd order statistics (fisher vectors)")
        print("[8] classification_method:")
        print("    'svm-linear': SVM with linear kernel")
        print("    'svm-rbf': SVM with RBF kernel")
        print("    'opf': OPF (supervised)")
        print("[9] output_suffix: Suffix to be added to the output files")
        print("#"*terminal_cols)
        sys.exit(-1)

    # read input parameters
    dataset_dirname = sys.argv[1]
    n_splits = sys.argv[2]
    use_samp_masks = sys.argv[3]
    int_point_detection_method = sys.argv[4]
    feat_extraction_method = sys.argv[5]
    dict_estimation_method = sys.argv[6]
    coding_method = sys.argv[7]
    classification_method = sys.argv[8]
    output_suffix = sys.argv[9]
    
    # set extra params
    save_patches = "0"
    save_dict_dataset = "0"
    sel_words_json = "-1"
    save_feat_vect = "0"
    perform_tsne = "0"

    # print input parameters
    print("#"*terminal_cols)
    print("Input parameters (iftBoVWLearnAndImgClassif.py script):")
    print("#"*terminal_cols)
    print("dataset_dirname:", dataset_dirname)
    print("n_splits:", n_splits)
    print("use_samp_masks:", use_samp_masks)
    print("int_point_detection_method:", int_point_detection_method)
    print("feat_extraction_method:", feat_extraction_method)
    print("dict_estimation_method:", dict_estimation_method)
    print("coding_method:", coding_method)
    print("classification_method:", classification_method)
    print("output_suffix:", output_suffix)

    output_dir_basename = "bovw_" + dataset_dirname
    output_dirname = output_dir_basename + '_' + int_point_detection_method + '_' + feat_extraction_method + '_' +
        dict_estimation_method + '_' + coding_method

    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)

    # create JSON with function params
    func_params = create_func_params_json(int_point_detection_method, feat_extraction_method, dict_estimation_method,
        coding_method, classification_method, use_samp_masks, patch_size, stride, col_space)

    # convert method names to numbers (according to   definition in LibIFT)
    [int_point_detection_method, feat_extraction_method, dict_estimation_method, coding_method,
        classification_method] = convert_method_names_to_numbers(int_point_detection_method, feat_extraction_method,
        dict_estimation_method, coding_method, classification_method)

    # execute the BoVW pipeline for each split
    for split in range(1, int(n_splits)+1):
        split = str(split)
        print("#"*terminal_cols)
        print("SPLIT: {} of {}".format(split, n_splits))
        print("#"*terminal_cols)

        # variables for each iteration
        filename_learn = os.path.join(dataset_dirname, "learn_" + str(split).zfill(3) + ".csv")
        filename_train = os.path.join(dataset_dirname, "train_" + str(split).zfill(3) + ".csv")
        filename_test = os.path.join(dataset_dirname, "test_" + str(split).zfill(3) + ".csv")

        output_suffix_per_split = output_suffix + "_split_" + str(split).zfill(3)
        input_bovw_dict = os.path.join(output_dirname, "dict_" + output_suffix_per_split + ".bovw")

        # set filenames for masks
        filename_learn_mask = "-1"
        if(use_samp_masks == "yes"):
            filename_learn_mask = os.path.join(dataset_dirname, "learn_" + str(split).zfill(3) + "_masks.csv")
        
        # execute iftBoVWLearn
        print("RUNNING iftBoVWLearn ...")
        print("#"*terminal_cols)
        cmd = "iftBoVWLearn {} {} {} {} {} {} {} {} {} {} {} {}".format(filename_learn, filename_learn_mask, int_point_detection_method,
            feat_extraction_method, dict_estimation_method, coding_method, json.dumps(func_params, sort_keys=False), save_patches,
            save_dict_dataset, sel_words_json, output_dir_basename, output_suffix_per_split)
        if (os.system(cmd) != 0):
                sys.exit("Error in iftBoVWLearn")

        # execute iftBoVWImgClassif
        print("RUNNING iftBoVWImgClassif ...")
        print("#"*terminal_cols)
        cmd = "iftBoVWImgClassif {} {} {} {} {} {}".format(input_bovw_dict, filename_train, filename_test, classification_method,
            save_feat_vect, perform_tsne,)
        if (os.system(cmd) != 0):
                sys.exit("Error in iftBoVWImgClassif")

    # create a JSON file with the results
    final_results_json = {}
    kappa_list, true_pos_list, true_pos_per_class_list, n_visual_words_list, n_feats_per_word_list = [], [], [], [], []
    dict_learn_time_list, feat_vect_extr_time_list, classif_time_list = [], [], []

    for split in range(1, int(n_splits)+1):
        # read the JSON file with the results
        output_suffix_per_split = output_suffix + "_split_" + str(split).zfill(3)
        results_filename = os.path.join(output_dirname, "results_" + output_suffix_per_split + ".json")
        fp = open(results_filename, 'r')
        results_json = json.load(fp)

        # add the results to the lists
        kappa_list.append(results_json["kappa"])
        true_pos_list.append(results_json["true_pos"])
        true_pos_per_class_list.append(results_json["true_pos_per_class"])
        n_visual_words_list.append(results_json["n_visual_words"])
        n_feats_per_word_list.append(results_json["n_feats_per_word"])
        dict_learn_time_list.append(results_json["dict_learn_time"])
        feat_vect_extr_time_list.append(results_json["feat_vect_extr_time"])
        classif_time_list.append(results_json["classif_time"])
    
    # compute mean and stdev
    final_results_json["int_point_detector"] = results_json["int_point_detector"]
    final_results_json["local_feat_extractor"] = results_json["local_feat_extractor"]
    final_results_json["dict_estimator"] = results_json["dict_estimator"]
    final_results_json["cod_func"] = results_json["cod_func"]
    final_results_json["classif_met"] = results_json["classif_met"]
    final_results_json["func_params"] = results_json["func_params"]
    final_results_json["n_classes"] = results_json["n_classes"]
    final_results_json["n_visual_words"] = "{:.2f} +- {:.2f}".format(np.mean(n_visual_words_list), np.std(n_visual_words_list))
    final_results_json["n_feats_per_word"] = "{:.2f} +- {:.2f}".format(np.mean(n_feats_per_word_list), np.std(n_feats_per_word_list))
    final_results_json["dict_learn_time"] = "{:.2f} +- {:.2f}".format(np.mean(dict_learn_time_list), np.std(dict_learn_time_list))
    final_results_json["feat_vect_extr_time"] = "{:.2f} +- {:.2f}".format(np.mean(feat_vect_extr_time_list), np.std(feat_vect_extr_time_list))
    final_results_json["classif_time"] = "{:.2f} +- {:.2f}".format(np.mean(classif_time_list), np.std(classif_time_list))
    final_results_json["kappa"] = "{:.2f} +- {:.2f}".format(np.mean(kappa_list)*100.0, np.std(kappa_list)*100.0)
    final_results_json["true_pos"] = "{:.2f} +- {:.2f}".format(np.mean(true_pos_list)*100.0, np.std(true_pos_list)*100.0)
    final_results_json["true_pos_per_class"] = "{} +- {}".format(list((np.mean(true_pos_per_class_list, axis=0)*100.0).round(decimals=2)),
                                                                 list((np.std(true_pos_per_class_list, axis=0)*100.0).round(decimals=2)))

    print("#"*terminal_cols)
    print("FINAL RESULTS ...")
    print("#"*terminal_cols)
    print("- dataset_dirname:", dataset_dirname)
    print("- n_splits:", n_splits)
    print("- use_samp_masks:", use_samp_masks)
    print("- int_point_detector: {}".format(final_results_json["int_point_detector"]))
    print("- local_feat_extractor: {}".format(final_results_json["local_feat_extractor"]))
    print("- dict_estimator: {}".format(final_results_json["dict_estimator"]))
    print("- cod_func: {}".format(final_results_json["cod_func"]))
    print("- classif_met: {}".format(final_results_json["classif_met"]))
    print("- func_params:")
    for p in func_params:
        print('  "', p, '": "', func_params[p], '"', sep='')
    print("- output_suffix:", output_suffix)
    print("- n_visual_words: {}".format(final_results_json["n_visual_words"]))
    print("- n_feats_per_word: {}".format(final_results_json["n_feats_per_word"]))
    print("- dict_learn_time: {}".format(final_results_json["dict_learn_time"]))
    print("- feat_vect_extr_time: {}".format(final_results_json["feat_vect_extr_time"]))
    print("- classif_time: {}".format(final_results_json["classif_time"]))
    print("- kappa: {}".format(final_results_json["kappa"]))
    print("- true_pos: {}".format(final_results_json["true_pos"]))
    print("- true_pos_per_class: {}".format(final_results_json["true_pos_per_class"]))

    final_results_filename = os.path.join(output_dirname, "results_" + output_suffix + ".json")
    fp = open(final_results_filename, 'w')
    json.dump(final_results_json, fp, sort_keys=False, indent=4)
