"""
Experiment script
"""

# Author: Samuel Martins <sbm.martins@gmail.com>
# License: Unicamp
# Date: April, 9, 2015

import NM
import classification as cl
import classification as cl
from sklearn.svm import SVC
import numpy as np
import os
import pdb
import random
import sys

from error import Error
import split


def validate_inputs(result_dir, log_filename, dataset_path, labels_path):
    """ Validate some inputs from the experiment.
        If the <result_dir: and the parent dir from the <log_filename> don't exist,
        they are created.
    """
    
    if os.path.exists(result_dir):
        if not os.path.isdir(result_dir):
            print("--- Creating the dir: \"%s\"\n" % result_dir)
            os.makedirs(result_dir)
    else:
        print("--- Creating the dir: \"%s\"\n" % result_dir)
        os.makedirs(result_dir)
    
    if not os.path.dirname(log_filename):
        print("--- Creating the dir: \"%s\"\n" % os.path.dirname(log_filename))
        os.makedirs(os.path.dirname(log_filename))

    if not os.path.exists(dataset_path):
        Error.print_error("Dataset file \"%s\" does not exist!" % dataset_path, "exp.validate_inputs")
    if not os.path.exists(labels_path):
        Error.print_error("Labels File \"%s\" does not exist!" % labels_path, "exp.validate_inputs")




def build_score_dict(neg_train_percs, max_NM_times, NM_methods):
    """Build dictionary to store the scores from the positive and negative classifications.
    E.g: neg_train_percs = [1., 2.]
         max_NM_times    = [2, 4, 6]
        
        return 
            pos_scores = {1: {2: [], 4: [], 6: []}, 2: {2: [], 4: [], 6: []}}
            neg_scores = {1: {2: [], 4: [], 6: []}, 2: {2: [], 4: [], 6: []}}

    Parameters
    ----------
    neg_train_percs: list of floats
        List of percentages of negative samples from the LARGE set that will
        be used as negative training in the Negative Mining.
    max_NM_times: list of floats
        List of maximum times for the Negative Mining.
    NM_methods: list of strings
        List of names of Negative Mining methods: "random, ou_NM, Felzs".
    """
    pos_scores = dict()
    neg_scores = dict()
    for neg_train_perc in neg_train_percs:
        pos_scores[neg_train_perc] = dict()
        neg_scores[neg_train_perc] = dict()
        for max_NM_time in max_NM_times:
            pos_scores[neg_train_perc][max_NM_time] = dict()
            neg_scores[neg_train_perc][max_NM_time] = dict()
            for NM_method in NM_methods:
                pos_scores[neg_train_perc][max_NM_time][NM_method] = []
                neg_scores[neg_train_perc][max_NM_time][NM_method] = []

    return pos_scores, neg_scores










def run_exp(params):
    params["result_dir"]   = os.path.abspath(os.path.expanduser(params["result_dir"]))
    params["log_filename"] = os.path.abspath(os.path.expanduser(params["log_filename"]))
    params["dataset"]      = os.path.abspath(os.path.expanduser(params["dataset"]))
    params["labels"]       = os.path.abspath(os.path.expanduser(params["labels"]))

    validate_inputs(params["result_dir"], params["log_filename"], params["dataset"],
        params["labels"])

    log = open(params["log_filename"], "w")
    log.write("************* NEGATIVE MINING EVALUATION *************\n")
    log.write("- Dataset: %s\n" % params["dataset"])
    log.write("- Labels: %s\n\n" % params["labels"])

    print("Loading Dataset")
    X = np.fromfile(params["dataset"], dtype="float32")
    print("Loading Labels")
    y = np.fromfile(params["labels"], dtype="int32")

    X.resize(y.size, X.size / y.size)
    log.write("X.shape: {0}\n".format(X.shape))


    neg_idxs, pos_train_idxs, pos_test_idxs = split.split_database(y, params["SPLIT"])

    log.write("Neg. set: {0}\n".format(neg_idxs["idxs"].size))
    log.write("Pos. Train. set: {0}\n".format(pos_train_idxs["idxs"].size))
    log.write("Pos. Test. set: {0}\n\n".format(pos_test_idxs["idxs"].size))

    # pos_scores, neg_scores = build_score_dict(params["NEG_MINING"]["neg_train_percs"],
    #                                           params["NEG_MINING"]["max_NM_times"],
    #                                           params["NEG_MINING"]["methods"].keys())
    
    # pdb.set_trace()

    i = 1
    pos_scores = dict()
    neg_scores = dict()
    for pos_label in set(pos_train_idxs["true_labels"]):
        log.write("--------------- {1}/{2} - Pos Label: {0} ---------------\n".format(pos_label,
                  i, len(np.unique(pos_train_idxs["true_labels"]))))

        # get the samples indices from X, whose label is "pos_label"
        pos_train_set     = pos_train_idxs["idxs"][np.where(pos_train_idxs["true_labels"]==pos_label)[0]]
        pos_test_set      = pos_test_idxs["idxs"][np.where(pos_test_idxs["true_labels"]==pos_label)[0]]
        impostor_test_set = set(pos_test_idxs["idxs"].tolist()) - set(pos_test_set.tolist())
        impostor_test_set = np.array(list(impostor_test_set))

        log.write("- Pos. Train. Set: {0}\n".format(len(pos_train_set)))
        log.write("- Pos. Test. Set: {0}\n\n".format(len(pos_test_set)))

        for neg_train_perc in params["NEG_MINING"]["neg_train_percs"]:
            pos_scores[neg_train_perc] = dict()
            neg_scores[neg_train_perc] = dict()

            # split randomly the LARGE negative set in training and validation
            neg_num            = int(neg_train_perc * len(neg_idxs["idxs"]))
            neg_train_set      = np.array(random.sample(neg_idxs["idxs"], neg_num))
            neg_validation_set = np.array(list(set(neg_idxs["idxs"]) - set(neg_train_set)))
            pdb.set_trace()


            for max_NM_time in params["NEG_MINING"]["max_NM_times"]:
                pos_scores[neg_train_perc][max_NM_time] = dict()
                neg_scores[neg_train_perc][max_NM_time] = dict()

                for NM_method in params["NEG_MINING"]["methods"]:
                    pos_scores[neg_train_perc][max_NM_time][NM_method] = []
                    neg_scores[neg_train_perc][max_NM_time][NM_method] = []

                    if NM_method == "random":
                        print("\n*********** RANDOM NEGATIVE MINING *************\n")
                        log.write("\n*********** RANDOM NEGATIVE MINING *************\n")
                        log.write("- Neg. Train. Perc.: {0}\n".format(neg_train_perc))
                        log.write("- Max. NM Time: {0}\n".format(max_NM_time))
                        log.write("***************************************************\n\n")
                        log.flush()

                        neg_mining_set = neg_train_set # we use the initial random neg. train. set
                    elif NM_method == "our_NM":
                        print("\n*********** OUR NEGATIVE MINING *************")
                        log.write("\n*********** OUR NEGATIVE MINING *************\n")
                        log.write("- Neg. Train. Perc.: {0}\n".format(neg_train_perc))
                        log.write("- Max. NM Time: {0}\n".format(max_NM_time))

                        neg_mining_set = NM.our_NM(X, neg_train_set, neg_validation_set,
                                            pos_train_set, max_NM_time, log)
                        log.write("***************************************************\n\n")
                        log.flush()
                        print("*********************************************************")

                    elif NM_method == "Felzenszwalb":
                        print("Felzenszwalb")
                    else:
                        Error.print_error("Invalid Negative Mining method: %s" % NM_method,
                            "exp.run_exp")

                    pos_scores[neg_train_perc][max_NM_time][NM_method], \
                    neg_scores[neg_train_perc][max_NM_time][NM_method] = \
                    cl.classify_by_SVM(X, pos_train_set, neg_mining_set,
                                       pos_test_set, impostor_test_set, log)
                    # pdb.set_trace()

        log.write("---------------------------------------------\n\n")

                    
        print ""



    log.close()






