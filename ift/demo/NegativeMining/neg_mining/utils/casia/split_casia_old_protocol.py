import numpy as np
import os
import pdb
import random
import sys


def print_error(error_msg, function):
    sys.exit("\nERROR in function {0}\n{1}".format(function, error_msg))





def validate_inputs(dataset_path, labels_path, out_dir, n_pos_labels, n_pos_imgs_per_label, pos_train_perc, n_neg_imgs_per_label):
    if not os.path.exists(dataset_path):
        print_error("\"%s\" does not exists" % dataset_path)

    if not os.path.exists(labels_path):
        print_error("\"%s\" does not exists" % labels_path)


    if os.path.exists(out_dir):
        if not os.path.isdir(out_dir):
            print_error("\"%s\" is not a directory" % out_dir)
    else:
        print("Creating the output directory \"%s\"\n" % out_dir)
        os.makedirs(out_dir)

    if n_pos_labels <= 0:
        print_error("Invalid n_pos_labels: %d... Try > 0" % n_pos_labels)

    if n_pos_imgs_per_label <= 0:
        print_error("Invalid n_pos_imgs_per_label: %d... Try > 0" % n_pos_imgs_per_label)

    if pos_train_perc <= 0:
        print_error("Invalid pos_train_perc: %d... Try > 0" % pos_train_perc)

    if n_neg_imgs_per_label <= 0:
        print_error("Invalid n_neg_imgs_per_label: %d... Try > 0" % n_neg_imgs_per_label)





def validate_sets(neg_idxs, pos_train_idxs, pos_test_idxs):
    ##################################
    # check the uniqueness from the idx sets
    if len(set(neg_idxs["ids"])) != len(neg_idxs["ids"]):
        print_error("Not Unique Negative Idxs", "validate_sets")
    if len(set(pos_train_idxs["ids"])) != len(pos_train_idxs["ids"]):
        print_error("Not Unique Pos. Train. Idxs", "validate_sets")
    if len(set(pos_test_idxs["ids"])) != len(pos_test_idxs["ids"]):
        print_error("Not Unique Pos. Test. Idxs", "validate_sets")

    if len(neg_idxs["ids"]) != len(neg_idxs["true_labels"]):
        print_error("Different sizes between Neg. idxs and Neg. true_labels", "validate_sets")
    if len(pos_train_idxs["ids"]) != len(pos_train_idxs["true_labels"]):
        print_error("Different sizes between Pos. Train. idxs and Pos. Train. true_labels", "validate_sets")
    if len(pos_test_idxs["ids"]) != len(pos_test_idxs["true_labels"]):
        print_error("Different sizes between Pos. Test. idxs and Pos. Test. true_labels", "validate_sets")

    if set(neg_idxs["ids"]).intersection(set(pos_train_idxs["ids"])) != set([]):
        print_error("Common idx(s) between Neg. Set and Pos. Train. Set", "validate_sets")
    if set(neg_idxs["ids"]).intersection(set(pos_test_idxs["ids"])) != set([]):
        print_error("Common idx(s) between Neg. Set and Pos. Test. Set", "validate_sets")
    if set(pos_train_idxs["ids"]).intersection(set(pos_test_idxs["ids"])) != set([]):
        print_error("Common idx(s) between Pos. Train. Set and Pos. Test. Set", "validate_sets")

    if set(neg_idxs["true_labels"]).intersection(set(pos_train_idxs["true_labels"])) != set([]):
        print_error("Common true_label(s) between Neg. Set and Pos. Train. Set", "validate_sets")
    if set(neg_idxs["true_labels"]).intersection(set(pos_test_idxs["true_labels"])) != set([]):
        print_error("Common true_label(s) between Neg. Set and Pos. Test. Set", "validate_sets")
    if set(pos_train_idxs["true_labels"]).intersection(set(pos_test_idxs["true_labels"])) != set(pos_test_idxs["true_labels"]):
        print_error("Different true_label(s) between Pos. Train. Set and Pos. Test. Set", "validate_sets")
    ##################################








def split_casia(y, n_pos_labels, n_pos_imgs_per_label, pos_train_perc, n_neg_imgs_per_label):
    neg_labels = []
    pos_labels = []

    sample_idxs = dict()

    # finds out the negative labels and the possible positive labels according to the n_pos_imgs_per_label
    for label in np.unique(y):
        sample_idxs[label] = np.where(y==label)[0]

        if sample_idxs[label].size < n_pos_imgs_per_label:
            neg_labels.append(label)
        else:
            pos_labels.append(label)

    if len(pos_labels) < n_pos_labels:
        print_error("Number of possible pos labels < n_pos_labels: %d - %d" % (len(pos_labels), n_pos_labels), "split_casia")

    # selects the pos. labels 
    random.shuffle(pos_labels)
    neg_labels.extend(pos_labels[n_pos_labels:])
    pos_labels = pos_labels[:n_pos_labels]
    if len(pos_labels) != n_pos_labels:
        print_error("Number of pos labels != n_pos_labels: %d - %d" % (len(pos_labels), n_pos_labels), "split_casia")

    # builds the negative set
    print("Builds the Negative Set")
    neg_idxs = dict()
    neg_idxs["ids"] = []
    neg_idxs["true_labels"] = []
    for label in neg_labels:
        if sample_idxs[label].size < n_neg_imgs_per_label:
            print_error("Num of images is < != n_neg_imgs_per_label: %d - %d" % (sample_idxs[label].size, n_neg_imgs_per_label), "split_casia")
        neg_idxs["ids"].extend(np.random.choice(sample_idxs[label], n_neg_imgs_per_label, replace=False))
        neg_idxs["true_labels"].extend([label] * n_neg_imgs_per_label)


    # builds the pos. train and test sets 
    print("Builds the Pos. Train and Test Set")
    pos_train_idxs = dict()
    pos_train_idxs["ids"] = []
    pos_train_idxs["true_labels"] = []
    pos_test_idxs  = dict()
    pos_test_idxs["ids"] = []
    pos_test_idxs["true_labels"] = []

    # builds the pos. train and test sets 
    pos_train_num = int(pos_train_perc*n_pos_imgs_per_label)
    for label in pos_labels:
        img_idxs = np.where(y==label)[0]
        img_idxs = np.random.choice(img_idxs, n_pos_imgs_per_label, replace=False)

        pos_train_idxs["ids"].extend(img_idxs[:pos_train_num])
        pos_train_idxs["true_labels"].extend([label]*pos_train_num)

        pos_test_idxs["ids"].extend(img_idxs[pos_train_num:])
        pos_test_idxs["true_labels"].extend([label]*(len(img_idxs)-pos_train_num))

    validate_sets(neg_idxs, pos_train_idxs, pos_test_idxs)

    neg_idxs["ids"]               = np.array(neg_idxs["ids"])
    neg_idxs["true_labels"]       = np.array(neg_idxs["true_labels"])
    pos_train_idxs["ids"]         = np.array(pos_train_idxs["ids"])
    pos_train_idxs["true_labels"] = np.array(pos_train_idxs["true_labels"])
    pos_test_idxs["ids"]          = np.array(pos_test_idxs["ids"])
    pos_test_idxs["true_labels"]  = np.array(pos_test_idxs["true_labels"])

    return neg_idxs, pos_train_idxs, pos_test_idxs



def save_feats(X, neg_idxs, pos_train_idxs, pos_test_idxs, out_dir):
    """Save a new dataset of features
    """
    ids    = np.concatenate((neg_idxs["ids"], pos_train_idxs["ids"], pos_test_idxs["ids"])) 
    labels = np.concatenate((neg_idxs["true_labels"], pos_train_idxs["true_labels"], pos_test_idxs["true_labels"])) 

    # save the new dataset
    print("Saving the New Dataset splitted")
    dataset_file = open(os.path.join(out_dir, "casia_new_feats.z"), "wb")
    for it, i in enumerate(ids):
        print("[%d/%d]" % (it+1, ids.size))
        X[i].tofile(dataset_file)
    dataset_file.close()

    # save the labels from the new dataset splitted
    print("Saving the New Label file with respect to the New Dataset")
    labels_file = open(os.path.join(out_dir, "casia_new_labels.z"), "wb")
    labels.tofile(labels_file)
    labels_file.close()

    # Save text files with the idxs of the set with respect to the new Dataset already splitted
    print("Saving the Neg. ids, Pos. train and test ids")
    new_neg_idxs       = np.arange(neg_idxs["ids"].size)
    new_pos_train_idxs = np.arange(neg_idxs["ids"].size, (neg_idxs["ids"].size+pos_train_idxs["ids"].size))
    new_pos_test_idxs  = np.arange((neg_idxs["ids"].size+pos_train_idxs["ids"].size), (neg_idxs["ids"].size+pos_train_idxs["ids"].size+pos_test_idxs["ids"].size))
    np.savetxt(os.path.join(out_dir, "neg_ids.txt"), new_neg_idxs, fmt="%d")
    np.savetxt(os.path.join(out_dir, "pos_train_ids.txt"), new_pos_train_idxs, fmt="%d")
    np.savetxt(os.path.join(out_dir, "pos_test_ids.txt"), new_pos_test_idxs, fmt="%d")






def main():
    if len(sys.argv) != 8:
        print_error("split_casia.py <dataset.z> <labels.z> <out_dir> <n_pos_labels> <n_pos_imgs_per_label> <pos_train_perc> <n_neg_imgs_per_label>", "main")

    dataset_path         = os.path.abspath(os.path.expanduser(sys.argv[1]))
    labels_path          = os.path.abspath(os.path.expanduser(sys.argv[2]))
    out_dir              = os.path.abspath(os.path.expanduser(sys.argv[3]))
    n_pos_labels        = int(sys.argv[4])
    n_pos_imgs_per_label = int(sys.argv[5])
    pos_train_perc       = float(sys.argv[6])
    n_neg_imgs_per_label = int(sys.argv[7])

    validate_inputs(dataset_path, labels_path, out_dir, n_pos_labels, n_pos_imgs_per_label, pos_train_perc, n_neg_imgs_per_label)

    print("Dataset: \"%s\"" % dataset_path)
    print("Labels: \"%s\"" % labels_path)
    print("Out Dir: \"%s\"" % out_dir)
    print("Number of Pos. Labels: %d" % n_pos_labels)
    print("Number of Imgs per Pos. Labels: %d" % n_pos_imgs_per_label)
    print("Pos. Train. Perc: %f" % pos_train_perc)
    print("Number of Imgs per Neg. Labels: %d\n" % n_neg_imgs_per_label)

    print("Loading Dataset")
    X = np.memmap(dataset_path, dtype="float32", mode="r")
    print("Loading Labels\n")
    y = np.fromfile(labels_path, dtype="int32")

    X.resize(y.size, X.size/y.size)

    neg_idxs, pos_train_idxs, pos_test_idxs = split_casia(y, n_pos_labels, n_pos_imgs_per_label, pos_train_perc, n_neg_imgs_per_label)

    save_feats(X, neg_idxs, pos_train_idxs, pos_test_idxs, out_dir)
    
    print("Done...\n")



if __name__ == "__main__":
    main()




