"""
Split script
"""

# Author: Samuel Martins <sbm.martins@gmail.com>
# License: Unicamp
# Date: April, 16, 2015

import numpy as np
import os
import pdb
import random
import sys

from error import Error


def validate_sets(neg_idxs, pos_train_idxs, pos_test_idxs):
    ##################################
    # check the uniqueness from the idx sets
    if len(set(neg_idxs["ids"])) != len(neg_idxs["ids"]):
        Error.print_error("Not Unique Negative Idxs", "split.validate_sets")
    if len(set(pos_train_idxs["ids"])) != len(pos_train_idxs["ids"]):
        Error.print_error("Not Unique Pos. Train. Idxs", "split.validate_sets")
    if len(set(pos_test_idxs["ids"])) != len(pos_test_idxs["ids"]):
        Error.print_error("Not Unique Pos. Test. Idxs", "split.validate_sets")

    if len(neg_idxs["ids"]) != len(neg_idxs["true_labels"]):
        Error.print_error("Different sizes between Neg. idxs and Neg. true_labels", "split.validate_sets")
    if len(pos_train_idxs["ids"]) != len(pos_train_idxs["true_labels"]):
        Error.print_error("Different sizes between Pos. Train. idxs and Pos. Train. true_labels", "split.validate_sets")
    if len(pos_test_idxs["ids"]) != len(pos_test_idxs["true_labels"]):
        Error.print_error("Different sizes between Pos. Test. idxs and Pos. Test. true_labels", "split.validate_sets")

    if set(neg_idxs["ids"]).intersection(set(pos_train_idxs["ids"])) != set([]):
        Error.print_error("Common idx(s) between Neg. Set and Pos. Train. Set", "split.validate_sets")
    if set(neg_idxs["ids"]).intersection(set(pos_test_idxs["ids"])) != set([]):
        Error.print_error("Common idx(s) between Neg. Set and Pos. Test. Set", "split.validate_sets")
    if set(pos_train_idxs["ids"]).intersection(set(pos_test_idxs["ids"])) != set([]):
        Error.print_error("Common idx(s) between Pos. Train. Set and Pos. Test. Set", "split.validate_sets")

    if set(neg_idxs["true_labels"]).intersection(set(pos_train_idxs["true_labels"])) != set([]):
        Error.print_error("Common true_label(s) between Neg. Set and Pos. Train. Set", "split.validate_sets")
    if set(neg_idxs["true_labels"]).intersection(set(pos_test_idxs["true_labels"])) != set([]):
        Error.print_error("Common true_label(s) between Neg. Set and Pos. Test. Set", "split.validate_sets")
    if set(pos_train_idxs["true_labels"]).intersection(set(pos_test_idxs["true_labels"])) != set(pos_test_idxs["true_labels"]):
        Error.print_error("Different true_label(s) between Pos. Train. Set and Pos. Test. Set", "split.validate_sets")
    ##################################


def split_dataset(y, split_params):
    """Split the dataset in Negative set, Pos. Train. and Test. sets.

    Parameters
    ----------
    y: numpy array of int
        List of labels, where each idx is the sample idx and its value is its label.
        E.g: y[500] = 4 --> the sample 500 has label 4

    split_params: dict of params
        n_neg_classes: int
            Number of Negative Classes from the dataset.
        n_pos_classes: int
            Number of Positive Classes from the dataset.
        n_imgs_per_class: int
            Number of Image per Class from the dataset.
        pos_train_perc: float
            Positive Training percentage.

    Returns
    -------
    neg_idxs: dict
        Dictionary with two keys:
            ["ids"] = numpy array of the idxs (with respect to y) from the negative samples.
            ["true_labels"] = numpy array of the true labels from the negative samples (with respect to y).

    pos_train_idxs: dict
            ["ids"] = numpy array of the idxs (with respect to y) from the pos. train. samples.
            ["true_labels"] = numpy array of the true labels from the pos. train. samples (with respect to y).
            e.g: pos_train_idxs["ids"][10] = 202
                 pos_train_idxs["true_labels"][10] = 80
                 it means that the pos. train. sample y[202] has label 80.

    pos_test_idxs: dict
            ["ids"] = numpy array of the idxs (with respect to y) from the pos. test. samples.
            ["true_labels"] = numpy array of the true labels from the pos. test. samples (with respect to y).
    """
    n_neg_classes    = split_params["n_neg_classes"]
    n_pos_classes    = split_params["n_pos_classes"]
    n_imgs_per_class = split_params["n_imgs_per_class"]
    pos_train_perc   = split_params["pos_train_perc"]

    unique_labels = list(set(y))
    sample_idxs = dict()

    for label in unique_labels:
        sample_idxs[label] = []

    # assign the idx to the respective labels
    for idx, label in enumerate(y):
        sample_idxs[label].append(idx)

    # get the negative and positive classes
    random.shuffle(unique_labels)
    neg_labels = unique_labels[:n_neg_classes]
    pos_labels = unique_labels[n_neg_classes:(n_neg_classes+n_pos_classes)]

    if set(neg_labels).intersection(set(pos_labels)) != set([]):
        Error.print_error("Duplicate some Negative and Positive Class", "split.split_dataset")

    # get the negative idxs
    neg_idxs = dict()
    neg_idxs["ids"] = []
    neg_idxs["true_labels"] = []
    for label in neg_labels:
        random.shuffle(sample_idxs[label])
        neg_idxs["ids"].extend(sample_idxs[label][:n_imgs_per_class])
        neg_idxs["true_labels"].extend([label] * len(sample_idxs[label][:n_imgs_per_class]))

    # get the positive idxs
    pos_train_idxs = dict()
    pos_train_idxs["ids"] = []
    pos_train_idxs["true_labels"] = []
    pos_test_idxs  = dict()
    pos_test_idxs["ids"] = []
    pos_test_idxs["true_labels"] = []
    for label in pos_labels:
        random.shuffle(sample_idxs[label])
        
        if n_imgs_per_class <= len(sample_idxs[label]):
            pos_train_num = int(pos_train_perc*n_imgs_per_class)

            pos_train_idxs["ids"].extend(sample_idxs[label][:pos_train_num])
            pos_train_idxs["true_labels"].extend([label] * len(sample_idxs[label][:pos_train_num]))

            pos_test_idxs["ids"].extend(sample_idxs[label][pos_train_num:n_imgs_per_class])
            pos_test_idxs["true_labels"].extend([label] * len(sample_idxs[label][pos_train_num:n_imgs_per_class]))
        else:
            print("-- WARNING - Label %d: Number of Samples < Number of Imgs Per Class - %d, :%d"
                  % (label, len(sample_idxs[label]), n_imgs_per_class))
            pos_train_num         = int(pos_train_perc*len(sample_idxs[label]))

            pos_train_idxs["ids"].extend(sample_idxs[label][:pos_train_num])
            pos_train_idxs["true_labels"].extend([label] * sample_idxs[label][:pos_train_num])

            pos_test_idxs["ids"].extend(sample_idxs[label][pos_train_num:len(sample_idxs[label])])
            pos_test_idxs["true_labels"].extend([label] * sample_idxs[label][pos_train_num:len(sample_idxs[label])])

    validate_sets(neg_idxs, pos_train_idxs, pos_test_idxs)

    neg_idxs["ids"]               = np.array(neg_idxs["ids"])
    neg_idxs["true_labels"]       = np.array(neg_idxs["true_labels"])
    pos_train_idxs["ids"]         = np.array(pos_train_idxs["ids"])
    pos_train_idxs["true_labels"] = np.array(pos_train_idxs["true_labels"])
    pos_test_idxs["ids"]          = np.array(pos_test_idxs["ids"])
    pos_test_idxs["true_labels"]  = np.array(pos_test_idxs["true_labels"])

    return neg_idxs, pos_train_idxs, pos_test_idxs





def split_database2(y, split_params):
    """Split the dataset in Negative set, Pos. Train. and Test. sets.

    Parameters
    ----------
    y: numpy array of int
        List of labels, where each idx is the sample idx and its value is its label.
        E.g: y[500] = 4 --> the sample 500 has label 4

    split_params: dict of params
        n_neg_classes: int
            Number of Negative Classes from the dataset.
        n_pos_classes: int
            Number of Positive Classes from the dataset.
        n_imgs_per_class: int
            Number of Image per Class from the dataset.
        pos_train_perc: float
            Positive Training percentage.

    Returns
    -------
    neg_idxs: dict
        Each key is a label and each value is a list of sample idxs from the negative samples.

    pos_train_idxs: dict
        Each key is a label and each value is a list of sample idxs from the pos. train. samples.

    pos_test_idxs: dict
        Each key is a label and each value is a list of sample idxs from the pos. test. samples.
    """
    n_neg_classes    = split_params["n_neg_classes"]
    n_pos_classes    = split_params["n_pos_classes"]
    n_imgs_per_class = split_params["n_imgs_per_class"]
    pos_train_perc   = split_params["pos_train_perc"]

    unique_labels = list(set(y))
    sample_idxs = dict()

    for label in unique_labels:
        sample_idxs[label] = []

    # assign the idx to the respective labels
    for idx, label in enumerate(y):
        sample_idxs[label].append(idx)

    # get the negative and positive classes
    random.shuffle(unique_labels)
    neg_labels = unique_labels[:n_neg_classes]
    pos_labels = unique_labels[n_neg_classes:(n_neg_classes+n_pos_classes)]

    if set(neg_labels).intersection(set(pos_labels)) != set([]):
        Error.print_error("Duplicate some Negative and Positive Class", "split.split_database2")

    # get the negative idxs
    neg_idxs = dict()
    for label in neg_labels:
        random.shuffle(sample_idxs[label])
        neg_idxs[label] = sample_idxs[label][:n_imgs_per_class]

    # get the positive idxs
    pos_train_idxs = dict()
    pos_test_idxs  = dict()
    for label in pos_labels:
        random.shuffle(sample_idxs[label])
        
        if n_imgs_per_class <= len(sample_idxs[label]):
            pos_train_num         = int(pos_train_perc*n_imgs_per_class)
            pos_train_idxs[label] = sample_idxs[label][:pos_train_num]
            pos_test_idxs[label]  = sample_idxs[label][pos_train_num:n_imgs_per_class]
        else:
            print("-- WARNING - Label %d: Number of Samples < Number of Imgs Per Class - %d, :%d"
                  % (label, len(sample_idxs[label]), n_imgs_per_class))
            pos_train_num         = int(pos_train_perc*len(sample_idxs[label]))
            pos_train_idxs[label] = sample_idxs[label][:pos_train_num]
            pos_test_idxs[label]  = sample_idxs[label][pos_train_num:len(sample_idxs[label])]

    ##################################
    # check the uniqueness from the idx
    all_neg_idxs = []
    all_pos_train_idxs = []
    all_pos_test_idxs = []
    for label, idxs in neg_idxs.items():
        all_neg_idxs += idxs
    for label, idxs in pos_train_idxs.items():
        all_pos_train_idxs += idxs
    for label, idxs in pos_test_idxs.items():
        all_pos_test_idxs += idxs

    if len(set(all_neg_idxs)) != len(all_neg_idxs):
        Error.print_error("Not Unique Negative Idxs", "split.split_database2")
    if len(set(all_pos_train_idxs)) != len(all_pos_train_idxs):
        Error.print_error("Not Unique Pos. Train. Idxs", "split.split_database2")
    if len(set(all_pos_test_idxs)) != len(all_pos_test_idxs):
        Error.print_error("Not Unique Pos. Test. Idxs", "split.split_database2")

    if set(all_neg_idxs).intersection(set(all_pos_train_idxs)) != set([]):
        Error.print_error("Common element(s) between Neg. Set and Pos. Train. Set", "split.split_database2")
    if set(all_neg_idxs).intersection(set(all_pos_test_idxs)) != set([]):
        Error.print_error("Common element(s) between Neg. Set and Pos. Test. Set", "split.split_database2")
    if set(all_pos_train_idxs).intersection(set(all_pos_test_idxs)) != set([]):
        Error.print_error("Common element(s) between Pos. Train. Set and Pos. Test. Set", "split.split_database2")
    ##################################

    return neg_idxs, pos_train_idxs, pos_test_idxs





def load_splitted_dataset(y, split_params):
    """Load the splitted sets: Negative set, Pos. Train. and Test. sets.

    Parameters
    ----------
    y: numpy array of int
        List of labels, where each idx is the sample idx and its value is its label.
        E.g: y[500] = 4 --> the sample 500 has label 4

    split_params: dict of params
        neg_set: string
            Path from the file which contains the neg set idxs.
        pos_train_set: string
            Path from the file which contains the pos. train. set idxs
        pos_test_set: string
            Path from the file which contains the pos. test. set idxs

    Returns
    -------
    neg_idxs: dict
        Dictionary with two keys:
            ["ids"] = numpy array of the idxs (with respect to y) from the negative samples.
            ["true_labels"] = numpy array of the true labels from the negative samples (with respect to y).

    pos_train_idxs: dict
            ["ids"] = numpy array of the idxs (with respect to y) from the pos. train. samples.
            ["true_labels"] = numpy array of the true labels from the pos. train. samples (with respect to y).
            e.g: pos_train_idxs["ids"][10] = 202
                 pos_train_idxs["true_labels"][10] = 80
                 it means that the pos. train. sample y[202] has label 80.

    pos_test_idxs: dict
            ["ids"] = numpy array of the idxs (with respect to y) from the pos. test. samples.
            ["true_labels"] = numpy array of the true labels from the pos. test. samples (with respect to y).
    """
    split_params["neg_set"]       = os.path.abspath(os.path.expanduser(split_params["neg_set"]))
    split_params["pos_train_set"] = os.path.abspath(os.path.expanduser(split_params["pos_train_set"]))
    split_params["pos_test_set"]  = os.path.abspath(os.path.expanduser(split_params["pos_test_set"]))

    neg_idxs = dict()
    neg_idxs["ids"]         = np.loadtxt(split_params["neg_set"], dtype="int")
    neg_idxs["true_labels"] = y[neg_idxs["ids"]]

    pos_train_idxs = dict()
    pos_train_idxs["ids"]         = np.loadtxt(split_params["pos_train_set"], dtype="int")
    pos_train_idxs["true_labels"] = y[pos_train_idxs["ids"]]

    pos_test_idxs = dict()
    pos_test_idxs["ids"]         = np.loadtxt(split_params["pos_test_set"], dtype="int")
    pos_test_idxs["true_labels"] = y[pos_test_idxs["ids"]]

    return neg_idxs, pos_train_idxs, pos_test_idxs







def split_CASIA_dataset(y, split_params):
    """Split the CASIA dataset in Negative set, Pos. Train. and Test. sets.
    The Negative set is built as follows:

    Zneg = Zneg1 + Zneg2
    
    Zneg1 = all images from the individuals which have less than <n_imgs_per_class>
    

    Z' = all images from the individuals which have at least <n_imgs_per_class>

    Zpos = Random <n_pos_classes> individuals from Z' with <n_imgs_per_class> chosen randomly.
    Zneg2 = all images from the other individuals (not positive) from Z'


    Parameters
    ----------
    y: numpy array of int
        List of labels, where each idx is the sample idx and its value is its label.
        E.g: y[500] = 4 --> the sample 500 has label 4

    split_params: dict of params
        n_pos_classes: int
            Number of Positive Classes from the dataset.
        n_imgs_per_class: int
            Number of Image per Class from the dataset.
        pos_train_perc: float
            Positive Training percentage.

    Returns
    -------
    neg_idxs: dict
        Dictionary with two keys:
            ["ids"] = numpy array of the idxs (with respect to y) from the negative samples.
            ["true_labels"] = numpy array of the true labels from the negative samples (with respect to y).

    pos_train_idxs: dict
            ["ids"] = numpy array of the idxs (with respect to y) from the pos. train. samples.
            ["true_labels"] = numpy array of the true labels from the pos. train. samples (with respect to y).
            e.g: pos_train_idxs["ids"][10] = 202
                 pos_train_idxs["true_labels"][10] = 80
                 it means that the pos. train. sample y[202] has label 80.

    pos_test_idxs: dict
            ["ids"] = numpy array of the idxs (with respect to y) from the pos. test. samples.
            ["true_labels"] = numpy array of the true labels from the pos. test. samples (with respect to y).
    """

    n_pos_labels     = split_params["n_pos_classes"]
    n_imgs_per_class = split_params["n_imgs_per_class"]
    pos_train_perc   = split_params["pos_train_perc"]

    # get the negative idxs
    neg_idxs = dict()
    neg_idxs["ids"] = []
    neg_idxs["true_labels"] = []

    valid_labels = []

    # get the individuals with less than <n_imgs_per_class>
    for label in np.unique(y):
        img_idxs = np.where(y==label)[0]
        # print("Label: %d - %d imgs" % (label, img_idxs.size))

        if img_idxs.size < n_imgs_per_class:
            neg_idxs["ids"].extend(img_idxs)
            neg_idxs["true_labels"].extend([label] * len(img_idxs))
        else:
            # add the possible positive individuals
            valid_labels.append(label)

    pos_train_idxs = dict()
    pos_train_idxs["ids"] = []
    pos_train_idxs["true_labels"] = []
    pos_test_idxs  = dict()
    pos_test_idxs["ids"] = []
    pos_test_idxs["true_labels"] = []

    random.shuffle(valid_labels)    
    pos_train_num = int(pos_train_perc*n_imgs_per_class)
    if n_pos_labels < len(valid_labels):
        # chosen positive indivuals
        for label in valid_labels[:n_pos_labels]:
            img_idxs = np.where(y==label)[0]
            img_idxs = np.random.choice(img_idxs, n_imgs_per_class, replace=False)

            pos_train_idxs["ids"].extend(img_idxs[:pos_train_num])
            pos_train_idxs["true_labels"].extend([label]*pos_train_num)
            pos_test_idxs["ids"].extend(img_idxs[pos_train_num:])
            pos_test_idxs["true_labels"].extend([label]*(len(img_idxs)-pos_train_num))

        # add all images from the remaining individuals to negative set
        for label in valid_labels[n_pos_labels:]:
            img_idxs = np.where(y==label)[0]
            neg_idxs["ids"].extend(img_idxs)
            neg_idxs["true_labels"].extend([label] * len(img_idxs))
    else:
        Error.print_error("Number of Positive Classes < Valid Classes", "split.split_CASIA_dataset")


    validate_sets(neg_idxs, pos_train_idxs, pos_test_idxs)

    neg_idxs["ids"]               = np.array(neg_idxs["ids"])
    neg_idxs["true_labels"]       = np.array(neg_idxs["true_labels"])
    pos_train_idxs["ids"]         = np.array(pos_train_idxs["ids"])
    pos_train_idxs["true_labels"] = np.array(pos_train_idxs["true_labels"])
    pos_test_idxs["ids"]          = np.array(pos_test_idxs["ids"])
    pos_test_idxs["true_labels"]  = np.array(pos_test_idxs["true_labels"])
    
    return neg_idxs, pos_train_idxs, pos_test_idxs












