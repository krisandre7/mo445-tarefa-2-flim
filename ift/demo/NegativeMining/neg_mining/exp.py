"""
Experiment script
"""

# Author: Samuel Martins <sbm.martins@gmail.com>
# License: Unicamp
# Date: April, 9, 2015


import numpy as np
import os
import pdb
import random
import sys
import tempfile

import classification as cl
from error import Error
import NM
from NM import CustomLinearSVM
import save
import split



def validate_inputs(result_dir, dataset_path, labels_path):
    """ Validate some inputs from the experiment.
        If the <result_dir>  doesn't exist, it is created.
    """
    
    if os.path.exists(result_dir):
        if not os.path.isdir(result_dir):
            print("--- Creating the dir: \"%s\"\n" % result_dir)
            os.makedirs(result_dir)
    else:
        print("--- Creating the dir: \"%s\"\n" % result_dir)
        os.makedirs(result_dir)
    
    if not os.path.exists(dataset_path):
        Error.print_error("Dataset file \"%s\" does not exist!" % dataset_path, "exp.validate_inputs")
    if not os.path.exists(labels_path):
        Error.print_error("Labels File \"%s\" does not exist!" % labels_path, "exp.validate_inputs")




def build_stats_dict(neg_train_percs, max_NM_times, NM_methods):
    """Build dictionary to store the scores from the positive and negative classifications, 
    and the processing time of each test.
    E.g: neg_train_percs = [1.0]
         max_NM_times    = [2, 4]
         NM_methods      = ["random_NM", "our_NM"]
        
        return 
            pos_scores = {1.0: {2: {"random_NM": [], "our_NM": []}, 4: {"random_NM": [], "our_NM": []}}}
            neg_scores = {1.0: {2: {"random_NM": [], "our_NM": []}, 4: {"random_NM": [], "our_NM": []}}}
            NM_proc_times = {1.0: {2: {"random_NM": [], "our_NM": []}, 4: {"random_NM": [], "our_NM": []}}}

    Parameters
    ----------
    neg_train_percs: list of floats
        List of percentages of negative samples from the LARGE set that will
        be used as negative training in the Negative Mining.
    max_NM_times: list of floats
        List of maximum times for the Negative Mining.
    NM_methods: list of strings
        List of names of Negative Mining methods: "random_NM, ou_NM, Felzs".
    """
    pos_scores = dict()
    neg_scores = dict()
    NM_proc_times = dict()
    for neg_train_perc in neg_train_percs:
        pos_scores[neg_train_perc] = dict()
        neg_scores[neg_train_perc] = dict()
        NM_proc_times[neg_train_perc] = dict()
        for max_NM_time in max_NM_times:
            pos_scores[neg_train_perc][max_NM_time] = dict()
            neg_scores[neg_train_perc][max_NM_time] = dict()
            NM_proc_times[neg_train_perc][max_NM_time] = dict()
            for NM_method in NM_methods:
                pos_scores[neg_train_perc][max_NM_time][NM_method] = []
                neg_scores[neg_train_perc][max_NM_time][NM_method] = []
                NM_proc_times[neg_train_perc][max_NM_time][NM_method] = []

    return pos_scores, neg_scores, NM_proc_times










def run_exp(params):
    params["result_dir"]   = os.path.abspath(os.path.expanduser(params["result_dir"]))
    params["log_filename"] = os.path.abspath(os.path.expanduser(params["log_filename"]))
    params["dataset"]      = os.path.abspath(os.path.expanduser(params["dataset"]))
    params["labels"]       = os.path.abspath(os.path.expanduser(params["labels"]))

    validate_inputs(params["result_dir"], params["dataset"], params["labels"])

    # log_filename = tempfile.mkstemp(dir=params["result_dir"], prefix="log_", suffix=".txt")[1]
    log = open(params["log_filename"], "w")
    log.write("************* NEGATIVE MINING EVALUATION *************\n")
    log.write("- Dataset: %s\n" % params["dataset"])
    log.write("- Labels: %s\n\n" % params["labels"])

    print("Loading Dataset")
    if (os.path.getsize(params["dataset"]) < params["memory_limit"]): 
        X = np.fromfile(params["dataset"], dtype="float32")
    else:
        X = np.memmap(params["dataset"], dtype="float32", mode="r")

    print("Loading Labels")
    y = np.fromfile(params["labels"], dtype="int32")

    X.resize(y.size, X.size / y.size)
    log.write("X.shape: {0}\n".format(X.shape))

    pos_scores, neg_scores, NM_proc_times = build_stats_dict(params["NEG_MINING"]["neg_train_percs"],
                                              params["NEG_MINING"]["max_NM_times"],
                                              params["NEG_MINING"]["methods"])


    ############## SPLIT DATASET ################
    split_methods = {"NORMAL": split.split_dataset, "FIXED": split.load_splitted_dataset, "CASIA": split.split_CASIA_dataset}
    chosen_split  = params["SPLIT"]["method"].upper() # chosen split method
    
    neg_set, pos_train_set, pos_test_set = split_methods[chosen_split](y, params["SPLIT"][chosen_split])
    
    if (params["SPLIT"]["save_splitted_set_ids"]):
        save.save_splitted_set_idxs(params["result_dir"], neg_set["ids"], 
             pos_train_set["ids"], pos_test_set["ids"])            
    ##############################################

    log.write("Neg. set: {0}\n".format(neg_set["ids"].size))
    log.write("Pos. Train. set: {0}\n".format(pos_train_set["ids"].size))
    log.write("Pos. Test. set: {0}\n\n".format(pos_test_set["ids"].size))
    log.flush()


    it = 1
    for pos_label in set(pos_train_set["true_labels"]):
        log.write("--------------- it: {1}/{2} - Pos Label: {0} ---------------\n".format(pos_label,
                  it, len(np.unique(pos_train_set["true_labels"]))))
        print("--------------- it: {1}/{2} - Pos Label: {0} ---------------".format(pos_label,
                  it, len(np.unique(pos_train_set["true_labels"]))))

        # get the sample indices from X, whose the label is "pos_label"
        pos_label_train_mask = pos_train_set["true_labels"]==pos_label
        pos_label_test_mask  = pos_test_set["true_labels"]==pos_label

        pos_train_ids        = pos_train_set["ids"][pos_label_train_mask]
        pos_test_ids         = pos_test_set["ids"][pos_label_test_mask]
        impostor_test_ids    = pos_test_set["ids"][~pos_label_test_mask]

        log.write("- Pos. Train. Set: {0}\n".format(len(pos_train_ids)))
        log.write("- Pos. Test. Set: {0}\n\n".format(len(pos_test_ids)))

        for neg_train_perc in params["NEG_MINING"]["neg_train_percs"]:
            # split randomly the LARGE negative set in training and validation
            neg_num            = int(neg_train_perc * len(neg_set["ids"]))
            neg_train_ids      = np.array(random.sample(neg_set["ids"], neg_num))
            neg_validation_ids = np.array(list(set(neg_set["ids"]) - set(neg_train_ids)))


            for max_NM_time in params["NEG_MINING"]["max_NM_times"]:
                for NM_method in params["NEG_MINING"]["methods"]:
                    custom_svm = None

                    if NM_method == "random_NM":
                        log.write("- Neg. Train. Perc.: {0}\n".format(neg_train_perc))
                        log.write("- Max. NM Time: {0}\n\n".format(max_NM_time))
                        log.flush()

                        custom_svm, spent_NM_time = NM.random_NM(X, neg_train_ids, 
                                            pos_train_ids, max_NM_time, params["NEG_MINING"]["normalize_data"], log)
                        neg_mining_set = neg_train_ids # we use the initial random neg. train. set
                    elif NM_method == "our_NM":
                        log.write("- Neg. Train. Perc.: {0}\n".format(neg_train_perc))
                        log.write("- Max. NM Time: {0}\n\n".format(max_NM_time))

                        custom_svm, spent_NM_time = NM.our_NM(X, neg_train_ids, neg_validation_ids,
                                            pos_train_ids, max_NM_time, params["NEG_MINING"]["normalize_data"], log)
                    elif NM_method == "Felzenszwalb_NM":
                        log.write("- Neg. Train. Perc.: {0}\n".format(neg_train_perc))
                        log.write("- Max. NM Time: {0}\n\n".format(max_NM_time))

                        custom_svm, spent_NM_time = NM.Felzenszwalb_NM(X, neg_train_ids, neg_validation_ids,
                                            pos_train_ids, max_NM_time, params["NEG_MINING"]["normalize_data"], log)
                    else:
                        Error.print_error("Invalid Negative Mining method: %s" % NM_method,
                            "exp.run_exp")
                    
                    print("- Final Classification")
                    p_scores, imp_scores = cl.classify_by_SVM(X, pos_test_ids, impostor_test_ids, custom_svm, log)
                    pos_scores[neg_train_perc][max_NM_time][NM_method].extend(p_scores)
                    neg_scores[neg_train_perc][max_NM_time][NM_method].extend(imp_scores)
                    NM_proc_times[neg_train_perc][max_NM_time][NM_method].append(spent_NM_time)

        it += 1
        log.write("---------------------------------------------\n\n")
        print("---------------------------------------------\n")
    log.write("**************************************************************************************************\n\n")

    save.save_scores(params["result_dir"], pos_scores, neg_scores)
    save.write_time_stats(NM_proc_times, log)
    log.close()






