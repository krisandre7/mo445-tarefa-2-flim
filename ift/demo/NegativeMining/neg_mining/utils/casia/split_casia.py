"""
Split dataset methods.
"""

# Author: Samuel Martins <sbm.martins@gmail.com>
# License: Unicamp
# Date: April, 9, 2015

import json
import numpy as np
import os
import pdb
import random
import re
import shutil
import sys
import time


def print_error(error_msg, function):
    sys.exit("\nERROR in function {0}\n{1}".format(function, error_msg))


def validate_sets(neg_ids, pos_train_ids, pos_test_ids):
    ##################################
    # check the uniqueness from the id sets
    if len(set(neg_ids)) != len(neg_ids):
        print_error("Not Unique Negative ids", "validate_sets")
    if len(set(pos_train_ids)) != len(pos_train_ids):
        print_error("Not Unique Pos. Train. ids", "validate_sets")
    if len(set(pos_test_ids)) != len(pos_test_ids):
        print_error("Not Unique Pos. Test. ids", "validate_sets")

    if set(neg_ids).intersection(set(pos_train_ids)) != set([]):
        print_error("Common id(s) between Neg. Set and Pos. Train. Set", "validate_sets")
    if set(neg_ids).intersection(set(pos_test_ids)) != set([]):
        print_error("Common id(s) between Neg. Set and Pos. Test. Set", "validate_sets")
    if set(pos_train_ids).intersection(set(pos_test_ids)) != set([]):
        print_error("Common id(s) between Pos. Train. Set and Pos. Test. Set", "validate_sets")
    ##################################



def split_CASIA_dataset(y, n_pos_labels, n_imgs_per_pos_label, pos_train_perc, n_neg_imgs, n_iters):
    """Split the CASIA dataset in Negative set, Pos. Train. and Test. sets.
    Firstly, choose <n_pos_labels> individuals from the ones that contain at least <n_imgs_per_pos_label> images
    to build the Positive Set (gallery).
    All remaining individuals and their images will build the huge negative set.
    Then, from this one, build <n_iters> large negative sets with <n_neg_imgs> images.


    Parameters
    ----------
    y: numpy array of int
        List of labels, where each idx is the sample idx and its value is its label.
        E.g: y[500] = 4 --> the sample 500 has label 4

    n_pos_labels: int
        Number of Positive Classes from the dataset.

    n_imgs_per_pos_label: int
        Number of Image per Class from the dataset.

    pos_train_perc: float
        Positive Training percentage.

    n_neg_imgs: int
        Number of Negative Images (Output Negative set size).

    n_iters: int
        Number of iterations

    Returns
    -------
    neg_ids: numpy array of shape (n_iters, n_neg_shapes)
        Numpy array of the idxs (with respect to y) from the negative samples.
        E.g: neg_ids[0] corresponds to the idxs of the iteration 0.

    pos_train_ids: numpy array of shape (n_imgs_per_pos_label*pos_train_perc)
        Numpy array of the idxs (with respect to y) from the pos. train. samples.

    pos_test_ids: numpy array of shape (n_imgs_per_pos_label*(1-pos_train_perc))
        Numpy array of the idxs (with respect to y) from the pos. test. samples.
    """
    # get the negative idxs
    huge_neg_ids = []
    valid_labels = []

    # get the individuals with less than <n_imgs_per_pos_label>
    for label in np.unique(y):
        img_idxs = np.where(y==label)[0]
        # print("Label: %d - %d imgs" % (label, img_idxs.size))

        if img_idxs.size < n_imgs_per_pos_label:
            huge_neg_ids.extend(img_idxs)
        else:
            # add the possible positive individuals
            valid_labels.append(label)

    pos_train_ids = []
    pos_test_ids  = []

    random.shuffle(valid_labels)    
    pos_train_num = int(pos_train_perc*n_imgs_per_pos_label)
    if n_pos_labels < len(valid_labels):
        # chosen positive indivuals
        for label in valid_labels[:n_pos_labels]:
            img_idxs = np.where(y==label)[0]
            img_idxs = np.random.choice(img_idxs, n_imgs_per_pos_label, replace=False)

            pos_train_ids.extend(img_idxs[:pos_train_num])
            pos_test_ids.extend(img_idxs[pos_train_num:])

        # add all images from the remaining individuals to negative set
        for label in valid_labels[n_pos_labels:]:
            img_idxs = np.where(y==label)[0]
            huge_neg_ids.extend(img_idxs)
    else:
        print_error("Number of Positive Classes < Valid Classes", "split_CASIA_dataset")

    # build the large negative sets randomly
    neg_ids = []
    for it in xrange(n_iters):
        random.shuffle(huge_neg_ids)
        neg_ids.append(huge_neg_ids[:n_neg_imgs])

        validate_sets(neg_ids[-1], pos_train_ids, pos_test_ids)

    return neg_ids, pos_train_ids, pos_test_ids


def save_sets(X, y, idxs, out_dataset, out_labelset):
    """Save the new dataset.

    Parameters
    ----------
    X: numpy array with shape (n_samples, n_feats)
        The original dataset.

    y: numpy array with shape (n_samples,)
        The label set.

    idxs: numpy array
        The indices from X and y of the chosen samples.

    out_dataset: string
        Output dataset pathname.

    out_labelset: string
        Output labelset pathname.
    """
    dataset_file = open(out_dataset, "wb")
    label_file   = open(out_labelset, "wb")
    # X_out = X[idxs]
    # y_out = y[idxs]

    # X_out.tofile(dataset_file)
    # y_out.tofile(label_file)

    # processes 100 samples per iteration
    n_samples = len(idxs)
    for i in xrange(0, n_samples, 100):
        print("\t- [%d/%d]" % (i, n_samples))    
        X_out = X[idxs[i:(i+100)]].flatten() # transforms the array to 1D
        X_out.tofile(dataset_file)

        y_out = y[idxs[i:(i+100)]].flatten() # transforms the array to 1D
        y_out.tofile(label_file)

    dataset_file.close()
    label_file.close()




def main():
    """Split the CASIA dataset into negative and positive set.
    Firstly, choose the gallery and negative users.
    Then, from the "huge negative set", build <n_iters> negative sets with <n_neg_imgs> negative images.
    """
    if (len(sys.argv) != 2):
        print_error("Usage: python split_dataset.py <split_params.json>\n", "main")

    json_file = open(sys.argv[1])
    try:
        params = json.load(json_file)
    except (ValueError, KeyError, TypeError):
        print_error("JSON format error", "main")

    dataset_path         = os.path.abspath(os.path.expanduser(params["dataset"]))
    labelset_path        = os.path.abspath(os.path.expanduser(params["labels"]))
    n_pos_labels         = params["n_pos_labels"]
    n_imgs_per_pos_label = params["n_imgs_per_pos_label"]
    pos_train_perc       = params["pos_train_perc"]
    n_neg_imgs           = params["n_neg_imgs"]
    n_iters              = params["n_iters"]
    out_dir              = os.path.abspath(os.path.expanduser(params["out_dir"]))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("- Loading Labels")
    y = np.fromfile(labelset_path, dtype="int32")

    print("- Loading Dataset: Feats")
    X = np.memmap(dataset_path, dtype="float32", mode="r")
    X = np.reshape(X, (y.size, X.size / y.size))

    print("- Spliting the dataset into Positive and Negative sets")
    # "ids" corresponds to the indices from the original dataset X and y
    neg_ids, pos_train_ids, pos_test_ids = split_CASIA_dataset(y, n_pos_labels, 
                                                    n_imgs_per_pos_label, pos_train_perc, n_neg_imgs, n_iters)
    
    # the indices from the pos and neg samples will be the same in all cutted datasets    
    # the negative sets of each iteration has the same size
    neg_idxs       = range(len(neg_ids[0]))
    pos_train_idxs = range(len(neg_ids[0]), len(neg_ids[0])+len(pos_train_ids))
    pos_test_idxs  = range(len(neg_ids[0])+len(pos_train_ids), len(neg_ids[0])+len(pos_train_ids)+len(pos_test_ids))

    print("- Saving Indices of the Cutted Dataset\n")
    np.savetxt(os.path.join(out_dir, "neg_ids.txt"), neg_idxs, fmt="%d")
    np.savetxt(os.path.join(out_dir, "pos_train_ids.txt"), pos_train_idxs, fmt="%d")
    np.savetxt(os.path.join(out_dir, "pos_test_ids.txt"), pos_test_idxs, fmt="%d")

    # Saving the cutted datasets - what changes are the negatives
    for it in xrange(n_iters):
        print("[%02d/%02d]" % ((it+1), n_iters))
        all_ids = neg_ids[it] + pos_train_ids + pos_test_ids

        out_dataset  = os.path.join(out_dir, "cut_casia_feats_it%02d.z" % (it+1))
        out_labelset = os.path.join(out_dir, "cut_casia_labels_it%02d.z" % (it+1))
        # save the cutted dataset
        print("- Saving dataset and labelset")
        save_sets(X, y, all_ids, out_dataset, out_labelset)


    print("\nDone...")


    # # pos_test_idxs = np.arange(len(neg_ids)+len(pos_train_ids), (len(neg_ids)+len(pos_train_ids)+len(pos_test_ids)))
    # # np.savetxt(os.path.join(out_dir, "pos_test_ids.txt"), pos_test_idxs, fmt="%d")

    # pdb.set_trace()    
    # # all_ids = neg_ids + pos_train_ids + pos_test_ids

    # # print("- Saving the new dataset")
    # # out_dataset  = os.path.join(out_dir, "cut_casia_feats.z")
    # # out_labelset = os.path.join(out_dir, "cut_casia_labels.z")
    # # save_sets(X, y, all_ids, out_dataset, out_labelset)

    # print("- Saving Positive Train and Test sets\n")
    # # "idxs" corresponds to the indices from the new dataset
    # neg_idxs = np.arange(len(neg_ids))
    # np.savetxt(os.path.join(out_dir, "neg_ids.txt"), neg_idxs, fmt="%d")
    
    # # pos_train_idxs = np.arange(len(neg_ids), (len(neg_ids)+len(pos_train_ids)))
    # # np.savetxt(os.path.join(out_dir, "pos_train_ids.txt"), pos_train_idxs, fmt="%d")

    # # pos_test_idxs = np.arange(len(neg_ids)+len(pos_train_ids), (len(neg_ids)+len(pos_train_ids)+len(pos_test_ids)))
    # # np.savetxt(os.path.join(out_dir, "pos_test_ids.txt"), pos_test_idxs, fmt="%d")




if __name__ == "__main__":
    sys.exit(main())
