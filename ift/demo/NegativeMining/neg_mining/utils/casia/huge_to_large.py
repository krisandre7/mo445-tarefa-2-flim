"""
Split dataset methods.

See a json file config demo in python ift/demo/NegativeMining/neg_mining/json/split_params.json
"""

# Author: Samuel Martins <sbm.martins@gmail.com>
# License: Unicamp
# Date: June, 16, 2015

import fnmatch
import json
import numpy as np
import os
import pdb
import random
import sys
import time

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

def print_error(error_msg, function):
    sys.exit("\nERROR in function {0}\n{1}".format(function, error_msg))


    


def validate_pos_and_neg_sets(pos_train_idxs, pos_test_idxs, neg_idxs):
    """Check if there are common samples (overlapping) among the pos train set, pos test set, and neg set."""

    if len(set(neg_idxs)) != len(neg_idxs):
        print_error("Not Unique Negative Idxs", "validate_pos_and_neg_sets")
    if len(set(pos_train_idxs)) != len(pos_train_idxs):
        print_error("Not Unique Pos. Train. Idxs", "validate_pos_and_neg_sets")
    if len(set(pos_test_idxs)) != len(pos_test_idxs):
        print_error("Not Unique Pos. Test. Idxs", "validate_pos_and_neg_sets")

    if set(neg_idxs).intersection(set(pos_train_idxs)) != set([]):
        print_error("Common idx(s) between Neg. Set and Pos. Train. Set", "validate_pos_and_neg_sets")
    if set(neg_idxs).intersection(set(pos_test_idxs)) != set([]):
        print_error("Common idx(s) between Neg. Set and Pos. Test. Set", "validate_pos_and_neg_sets")
    if set(pos_train_idxs).intersection(set(pos_test_idxs)) != set([]):
        print_error("Common idx(s) between Pos. Train. Set and Pos. Test. Set", "validate_pos_and_neg_sets")



def save_idx_set(idxs, filename):
    """Save the indice set to the file <filename>.
    """
    np.savetxt(filename, idxs, fmt="%d")




def split_dataset(y, n_pos_labels, n_imgs_per_label, pos_train_perc):
    """Split the dataset into positive training set, positive testing set and negative set.

    Parameters
    ----------
    y: numpy array with shape (n_samples,)
        Label set.

    n_pos_labels: int
        Number of positive labels required.

    n_imgs_per_label: int
        Number of images per individual/label required in the positive set.

    pos_train_perc: float in [0..1]
        Training Percentage from the positive training and testing set.

    Returns
    -------
    pos_train_idxs: numpy array with shape (n_pos_labels*n_imgs_per_label*pos_train_perc)
        Set of indices from the chosen positive training samples.

    pos_test_idxs: numpy array with shape (n_pos_labels*n_imgs_per_label*(1-pos_train_perc))
        Set of indices from the chosen positive testing samples.

    neg_idxs: numpy array
        Set of indices from the chosen negative samples.
    """
    unique_labels = np.unique(y)
    sample_idxs   = dict()

    # initialize the dict
    for label in unique_labels:
        sample_idxs[label] = []

    # assign the idx to the respective labels
    for idx, label in enumerate(y):
        sample_idxs[label].append(idx)

    # get the negative and positive classes
    random.shuffle(unique_labels)
    pos_labels = unique_labels[:n_pos_labels]
    neg_labels = unique_labels[n_pos_labels:]

    if set(neg_labels).intersection(set(pos_labels)) != set([]):
        print_error("Duplicate some Negative and Positive Class", "split.split_dataset")

    # get the positive idxs
    pos_train_idxs = []
    pos_test_idxs  = []

    for label in pos_labels:
        random.shuffle(sample_idxs[label])
        
        if n_imgs_per_label <= len(sample_idxs[label]):
            pos_train_num = int(pos_train_perc*n_imgs_per_label)

            pos_train_idxs.extend(sample_idxs[label][:pos_train_num])
            pos_test_idxs.extend(sample_idxs[label][pos_train_num:n_imgs_per_label])
        else:
            print_error("-- WARNING - Label %d: Number of Samples < Number of Imgs Per Class - %d, :%d"
                  % (label, len(sample_idxs[label]), n_imgs_per_label), "split_dataset")

    # get the negative idxs
    neg_idxs = []
    for label in neg_labels:
        random.shuffle(sample_idxs[label])
        neg_idxs.extend(sample_idxs[label])


    validate_pos_and_neg_sets(pos_train_idxs, pos_test_idxs, neg_idxs)


    return np.array(pos_train_idxs), np.array(pos_test_idxs), np.array(neg_idxs)



def build_label_dict(y, idxs):
    """Build a dict with the idxs from y of each label.


    Parameters
    ----------
    y: numpy array with shape (n_samples,)
        Label set.

    idxs: numpy array
        Set with the indices of negative samples previously chosen.
        These indices are related with y.


    Returns
    -------
    sample_idxs: dict
        Dictionary where each key corresponds to a label that stores the idxs from the sample of that label.
        The idxs are realeted to y.
    """
    unique_labels = np.unique(y[idxs]) # get all labels from the negatives
    sample_idxs   = dict()

    # initialize the dict
    for label in unique_labels:
        sample_idxs[label] = []

    # assign the idx to its respective label
    for idx in idxs:
        sample_idxs[y[idx]].append(idx)

    return sample_idxs




def random_neg_set(idxs, met_params):
    """Split the Negative Set by choosing <n_neg_imgs> randomly.
    
    Parameters
    ----------
    idxs: numpy array
        Set with the indices of negative samples previously chosen.

    met_params: dict
        "n_neg_imgs": int
            Number of negative images required.

        "neg_set_fname": string
            Filename which will be used to save the neg set.

    Notes
    -----
    If the number of images required is greater than the total number of images, an error message will be printed
    and the program will be terminated.
    """
    n_neg_imgs  = met_params["n_neg_imgs"]
    chosen_idxs = None

    if n_neg_imgs <= idxs.size:
        idxs_aux = np.array(idxs) 
        np.random.shuffle(idxs_aux)
        chosen_idxs = idxs_aux[:n_neg_imgs]
    else:
        print_error("Number of neg imgs required is > than the neg set: %d -- %d" % (n_neg_imgs, idxs.size),
            "random_neg_set")

    return chosen_idxs


def random_pl_neg_set(y, idxs, met_params):
    """Split the Negative Set by choosing <n_neg_imgs_per_label> images per label/class randomly.
    
    Parameters
    ----------
    y: numpy array with shape (n_samples,)
        Label set.

    idxs: numpy array
        Set with the indices of negative samples previously chosen.
        These indices are related with y.

    met_params: dict
        "n_neg_imgs_per_label": int
            Number of negative images per class required.

        "neg_set_fname": string
            Filename which will be used to save the neg set.

    Notes
    -----
    If the number of images per class required is greater than the total number of images per class, 
    an error message will be printed and the program will be terminated.
    """
    n_neg_imgs_per_label = met_params["n_neg_imgs_per_label"]

    sample_idxs = build_label_dict(y, idxs)

    chosen_idxs = []

    for cluster in sample_idxs:
        if n_neg_imgs_per_label <= len(sample_idxs[cluster]):
            np.random.shuffle(sample_idxs[cluster])
            chosen_idxs.extend(sample_idxs[cluster][:n_neg_imgs_per_label])
        else:
            print_error("Insufficient number of imgs from the label %d: %d imgs" % (cluster, len(sample_idxs[cluster])),
                "random_pl_neg_set")
    
    chosen_idxs = np.array(chosen_idxs)

    return chosen_idxs



def kmeans_neg_set(y, idxs, met_params):
    """Split the Negative Set by kmeans.
    Clusters each individual/label in <n_clusters> clusters and selects <n_neg_imgs_per_label> images
    per label/class/individual following the idea:
        - Get the nearest elements from each centroid until reaching <n_neg_imgs_per_label>.
    
    Parameters
    ----------   
    y: numpy array with shape (n_samples,)
        Label set.

    idxs: numpy array
        Set with the indices of negative samples previously chosen.
        These indices are related with y.

    met_params: dict
        "dataset": string
            Dataset with the features which will be used for clustering.

        "n_clusters": int
            Number of clusters that will be used.

        "n_neg_imgs_per_label": int
            Number of negative images per individual/label required.

        "neg_set_fname": string
            Filename which will be used to save the neg set.

    Notes
    -----
    If the number of images per class required is greater than the total number of images per class, 
    an error message will be printed and the program will be terminated.
    """
    dataset_path         = os.path.abspath(os.path.expanduser(met_params["dataset"]))
    n_clusters           = met_params["n_clusters"]
    n_neg_imgs_per_label = met_params["n_neg_imgs_per_label"]

    print("\t* dataset_path = %s" % dataset_path)
    print("\t* n_neg_imgs_per_label = %s" % n_neg_imgs_per_label)
    print("\t* n_clusters = %d\n" % n_clusters)

    sample_idxs = build_label_dict(y, idxs)

    print("\t- Loading the dataset")
    X = np.fromfile(dataset_path, dtype="float32")
    X = np.reshape(X, (y.size, X.size/y.size))
    
    print("\t- Clustering selection")    
    chosen_idxs = []

    for label, orig_label_idxs in sample_idxs.items():
        print("\t[individual: %d]" % label)

        X_train = X[orig_label_idxs]
        # print("\t- Normalizing the samples")
        X_train = normalize(X_train, norm="l2") # normalize the dataset
        kmeans  = KMeans(n_clusters=n_clusters)

        # print("\t- Clustering the dataset")
        centroid_dists = kmeans.fit_transform(X_train) # cluster X_train and get the distances from each sample to each centroid

        ###############################################
        # builds a dict with each key corresponds to a cluster that stores the id the samples (with respect to y/X) and
        # the distance to their respective centroid
        cluster_sample_dists = dict()

        for cluster in np.unique(kmeans.labels_):
            cluster_sample_dists[cluster] = []

        for sample in xrange(kmeans.labels_.size): # for each sample
            cluster   = kmeans.labels_[sample]
            sample_id = orig_label_idxs[sample] # index with respect to y/X and not X_train
            dist      = centroid_dists[sample][cluster] # distance from the "sample" to its respective cluster

            cluster_sample_dists[cluster].append((sample_id, dist))
        
        # sorts (in place) in ascending order the samples according to their distances to centroid
        for cluster in np.unique(kmeans.labels_):
            cluster_sample_dists[cluster].sort(key=lambda tup: tup[1])
            # print("\t\t- cluster: %d - %d samples" % (cluster, len(cluster_sample_dists[cluster])))
    # print("")
    ###############################################
        
        ###############################################
        # Selects the representative samples from the individual <label>
        rep_samples = [] # stores the indices with respect to y/X from the representative samples from the <label>
        
        cluster = 0
        while len(rep_samples) < n_neg_imgs_per_label:    
            cluster = cluster % n_clusters # circular cluster selection
            
            if len(cluster_sample_dists[cluster]) >= 1:
                # print("\t- cluster = %d" % cluster)
                sample_id, dist = cluster_sample_dists[cluster].pop(0) # gets and removes the key cluster from the dict
                rep_samples.append(sample_id)
            cluster += 1
        
        chosen_idxs.extend(rep_samples)
        ###############################################

    del X
    
    chosen_idxs = np.array(chosen_idxs)
    
    return chosen_idxs





def main():
    """Split the CASIA dataset into negative and positive set, according to some strategies.
    The dataset used MUST BE BALANCED, i. e., it must have the same number of images per class (see the script: cut_casia.py),
    in order to get more confident results.
    
    The script can execute several negative splits, but the positive set will be the same for everyone.
    """
    if (len(sys.argv) != 2):
        print_error("Usage: python split_dataset.py <split_params.json>\n", "main")

    json_file = open(sys.argv[1])
    try:
        params = json.load(json_file)
    except (ValueError, KeyError, TypeError):
        print_error("JSON format error", "main")

    t1 = time.time() # points the time
    print("- Loading Labels")
    params["labels"] = os.path.abspath(os.path.expanduser(params["labels"]))
    y = np.fromfile(params["labels"], dtype="int32")

    out_dir = os.path.abspath(os.path.expanduser(params["out_dir"]))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    n_iters = params["n_iters"]

    for it in xrange(1, n_iters+1):
        print("- Spliting the dataset into Positive and Negative sets")
        pos_train_idxs, pos_test_idxs, neg_idxs = split_dataset(y, params["n_pos_labels"], params["n_imgs_per_label"], params["pos_train_perc"])
        
        print("- Saving Positive Train and Test sets\n")
        pos_train_set_fname = os.path.join(out_dir, ("pos_train_ids-it%02d.txt" % it))
        save_idx_set(pos_train_idxs, pos_train_set_fname)
        pos_test_set_fname = os.path.join(out_dir, ("pos_test_ids-it%02d.txt" % it))
        save_idx_set(pos_test_idxs, pos_test_set_fname)
        
        for method, met_params in params["METHODS"].items():
            chosen_idxs = None

            if method == "RANDOM":
                print("- Random Negative selection")
                chosen_idxs = random_neg_set(neg_idxs, met_params)
            elif method == "RANDOM_PER_LABEL":
                print("- Random (Per label) Negative selection")
                chosen_idxs = random_pl_neg_set(y, neg_idxs, met_params)
            elif fnmatch.fnmatch(method, "KMEANS-*"):
                print("- Kmeans Negative Selection")
                chosen_idxs = kmeans_neg_set(y, neg_idxs, met_params)
            else:
                print("Invalid Sample Selection method: %s" % method)

            if chosen_idxs is not None:
                neg_set_fname = met_params["exp_name"] + ("-it%02d.txt" % it)
                neg_set_fname = os.path.join(out_dir, neg_set_fname) 
                print(neg_set_fname)
                save_idx_set(chosen_idxs, neg_set_fname)
                print("")

    print("\nTime elapsed %.2f secs" % (time.time() - t1))
    print("Done...")



if __name__ == "__main__":
    sys.exit(main())
