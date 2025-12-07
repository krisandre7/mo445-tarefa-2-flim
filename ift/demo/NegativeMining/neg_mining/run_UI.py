"""
Run User-Independent algorithms
"""
# Author: Samuel Martins <sbm.martins@gmail.com>
# License: Unicamp
# Date: April, 9, 2015

# See an example of json plot config in "neg_mining/plot/UI_configs.json"

import json
import os
import pdb
import sys
import time

import numpy as np
import numpy.linalg as LA
from scipy.spatial.distance import cosine as cos_dist
from sklearn.decomposition import PCA
from sklearn.lda import LDA

import split

def print_error(msg, function):
    sys.exit("Error in {0} - {1}".format(function, msg))


def validate_inputs(dataset_path, labels_path, pos_score_path, neg_score_path):
    """ Validate some inputs from the experiment.
    """   
    if not os.path.exists(dataset_path):
       print_error("Dataset file \"%s\" does not exist!" % dataset_path, "run_UI.validate_inputs")
    if not os.path.exists(labels_path):
       print_error("Labels File \"%s\" does not exist!" % labels_path, "run_UI.validate_inputs")

    parent_dir = os.path.dirname(pos_score_path)
    if not os.path.isdir(parent_dir):
        print("--- Creating the dir: \"%s\"\n" % parent_dir)
        os.makedirs(parent_dir)

    parent_dir = os.path.dirname(neg_score_path)
    if not os.path.isdir(parent_dir):
        print("--- Creating the dir: \"%s\"\n" % parent_dir)
        os.makedirs(parent_dir)




def apply_1NN(train_set, train_labels, test_set, test_labels):
    train_norm = LA.norm(train_set, axis=1)
    test_norm  = LA.norm(test_set, axis=1)

    # normalizing the datasets for applying cosine similarity
    train_set2 = np.array([train_set[i]/train_norm[i] for i in xrange(len(train_set))])
    test_set2  = np.array([test_set[i]/test_norm[i] for i in xrange(len(test_set))])

    # to avoid approximation noise in unit vector check
    train_norm = np.array([int(norm+10e-6) for norm in LA.norm(train_set2, axis=1)])
    test_norm  = np.array([int(norm+10e-6) for norm in LA.norm(test_set2, axis=1)])
    # pdb.set_trace()

    # Check if all samples from train_set2 and test_set2 have 1-norm
    if not all(train_norm == np.array([1]*len(train_norm))):
        print_error("Some sample from the Normalized Training Set does not have 1-norm", "apply_1NN")
    
    if not all(test_norm == np.array([1]*len(test_norm))):
        print_error("Some sample from the Normalized Testing Set does not have 1-norm", "apply_1NN")

    # pdb.set_trace()
    
    negatives = []
    positives = []

    total = len(test_set2) * len(train_set2)
    for s, test in enumerate(test_set2):
        scores = {} # each train label has a list of scores (cos_sim) for the sample <test>
        for label in list(set(train_labels)): # for each train label
            scores[label] = []
        # pdb.set_trace()

        # train_set2.shape = (n_samples, n_comps), test.shape = (n_comps,)
        score_matrix = np.dot(train_set2, test) # shape = (n_samples,)
        
        for t in xrange(score_matrix.shape[0]):
            scores[train_labels[t]].append(score_matrix[t])

        # Get the highest cosine similarity from each training individual to <test>
        pos_label = test_labels[s]
        positives.append(max(scores[pos_label]))

        neg_labels = scores.keys()
        neg_labels.remove(pos_label)
        for nl in neg_labels:
            negatives.append(max(scores[nl]))
        # pdb.set_trace()

    return np.array(positives), np.array(negatives)












def run_User_Independent(params):
    params["dataset"]                   = os.path.abspath(os.path.expanduser(params["dataset"]))
    params["labels"]                    = os.path.abspath(os.path.expanduser(params["labels"]))
    params["save"]["pos_scores"] = os.path.abspath(os.path.expanduser(params["save"]["pos_scores"]))
    params["save"]["neg_scores"] = os.path.abspath(os.path.expanduser(params["save"]["neg_scores"]))


    validate_inputs(params["dataset"], params["labels"], params["save"]["pos_scores"], params["save"]["neg_scores"])

    print("- Loading Dataset")
    if (os.path.getsize(params["dataset"]) < params["memory_limit"]): 
        X = np.fromfile(params["dataset"], dtype="float32")
    else:
        X = np.memmap(params["dataset"], dtype="float32", mode="r")

    print("- Loading Labels\n")
    y = np.fromfile(params["labels"], dtype="int32")

    X.resize(y.size, X.size / y.size)

    neg_set, pos_train_set, pos_test_set = split.load_splitted_dataset(y, params["SPLIT"])

    print("- X.shape: {0}".format(X.shape))
    print("- y.shape: {0}".format(y.shape))
    print("- neg_set.shape: {0}".format(neg_set["ids"].shape))
    print("- pos_train_set.shape: {0}".format(pos_train_set["ids"].shape))
    print("- pos_test_set.shape: {0}\n".format(pos_test_set["ids"].shape))


    X_neg = X[neg_set["ids"]]
    # "gambiarra" to work - the amount of samples is exploding the fit
    X_neg = X_neg[:10500]

    for method in params["method"]:
        print("************************* %s *************************" % method)

        proj = None        
        if (method == "PCA"):
            print("- Learning PCA subspace")
            n_comps     = min(X_neg.shape)
            proj        = PCA(n_components=n_comps)
            # just to call an only fit for PCA and LDA
            # label is not used in PCA fit
            y_neg = None
        elif (method == "LDA"):
            print("- Learning LDA subspace")
            n_comps = len(np.unique(neg_set["true_labels"]))-1 # n_neg_classes-1
            proj    = LDA(n_components=n_comps)
            y_neg   = neg_set["true_labels"] # just to use an only fit for PCA and LDA
            # "gambiarra" to work - the amount of samples is exploding the fit
            y_neg = y_neg[:10500]
        else:
            print_error("Invalid method: \"%s\"" % method, "run_UI.run_User_Independent")
        
        t_fit = time.time()
        proj.fit(X=X_neg, y=y_neg)
        print("--> Time elapsed - Fit the model: {0:.2f} secs\n".format(time.time()-t_fit))

        print("- Projecting Positive Training Samples on subspace")
        X_pos_train = proj.transform(X[pos_train_set["ids"]])
        y_pos_train = pos_train_set["true_labels"]

        print("- Projecting Positive Testing Samples on subspace")
        X_pos_test = proj.transform(X[pos_test_set["ids"]])
        y_pos_test = pos_test_set["true_labels"]
    
        t_cl = time.time()
        print("\n- Classifying Positives")
        positives, negatives = apply_1NN(X_pos_train, y_pos_train, X_pos_test, y_pos_test)
        print("--> Time elapsed - Classifying Positives: {0:.2f} secs\n".format(time.time()-t_cl)) 

        ################### SAVING THE RESULTING SCORES #########################
        print("........... Saving Positive scores for ROC curve ...........")
        print("- positives.size = %d" % len(positives))
        print("- negatives.size = %d\n" % len(negatives))

        if not params["save"]["pos_scores"].endswith(".npy"):
            print_error("Numpy extension .npy not found in \"%s\"" % params["save"]["pos_scores"],
                        "run_UI.run_User_Independent")
        filename = params["save"]["pos_scores"].split(".npy")[0] + "." + method + ".npy"
        print("- Pos score filename: \"%s\"" % filename)
        np.save(filename, positives)

        if not params["save"]["neg_scores"].endswith(".npy"):
            print_error("Numpy extension .npy not found in \"%s\"" % params["save"]["neg_scores"],
                        "run_UI.run_User_Independent")
        filename = params["save"]["neg_scores"].split(".npy")[0] + "." + method + ".npy"
        print("- Neg score filename: \"%s\"" % filename)
        np.save(filename, negatives)
        print("*******************************************************\n")







def main():
    if (len(sys.argv) == 1):
        print_error("run_UI.py <params01.json> <params02.json> ...", "main")
        
    t1 = time.time()
    for i in range(1, len(sys.argv)):
        json_file = open(sys.argv[i])

        try:
            configs = json.load(json_file)
        except (ValueError, KeyError, TypeError):
            print_error("JSON format error", "main")

        run_User_Independent(configs)

        json_file.close()
        
    print("\n********* Total Time elapsed: {0:.2f} secs *********".format(time.time()-t1))
    print("\nDone...")
    
    
if __name__ == "__main__":
    sys.exit(main())
