import numpy as np
import os
import pdb
import sys
import time



def print_error(error_msg, function):
    sys.exit("\nERROR in function {0}\n{1}".format(function, error_msg))


def validate_inputs(dataset_path, labels_path, n_imgs, out_dir):
    if not os.path.exists(dataset_path):
        print_error("Dataset \"%s\" does not exist!" % dataset_path, "validate_inputs")

    if not os.path.exists(labels_path):
        print_error("Labels \"%s\" does not exist!" % labels_path, "validate_inputs")

    if n_imgs <= 0:
        print_error("Invalid Minimum Number of Images per Individiual: %d... Try > 0" % n_imgs, "validate_inputs")

    if os.path.exists(out_dir):
        if not os.path.isdir(out_dir):
            msg = "Out dir is a file and not a dir: %s" % out_dir
            print_error(msg, "validate_inputs")
    else:
        print("- Creating the output dir: %s\n" % out_dir)
        os.makedirs(out_dir)




def cut_dataset(y, n_min_imgs):
    """Selects the individuals with at least <n_min_imgs> images.
    After that, it selects <n_min_imgs> random images from each one.

    Parameters
    ----------
    y: numpy array
        Numpy Array with the labels of the images

    n_min_imgs: int
        Minimum number of images that an individual must have.

    Returns
    -------
    idxs: numpy array
        Numpy array that stores the indices from the chosen samples.
        These indices are related to the original dataset.

    unchosen_idxs: numpy array
        Numpy array that stores the indices from the unchosen samples.
        These indices are related to the original dataset.
    """
    idxs          = []
    unchosen_idxs = []

    # get only the individuals with at least <n_min_imgs>
    for label in np.unique(y):
        img_idxs = np.where(y==label)[0]
        # print("Label: %d - %d imgs" % (label, img_idxs.size))

        if img_idxs.size >= n_min_imgs:
            img_idxs = np.random.choice(img_idxs, n_min_imgs)
            idxs.extend(img_idxs)
        else:
            unchosen_idxs.extend(img_idxs)

    return np.array(idxs), np.array(unchosen_idxs)



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
    X_out = X[idxs]
    y_out = y[idxs]

    X_out.tofile(dataset_file)
    y_out.tofile(label_file)

    dataset_file.close()
    label_file.close()



def main():
    """Slices the Casia WebFace dataset.

    Build a new dataset with the individuals which have at least <n_imgs> images.
    After that, choose <n_imgs> random for each one.
    """
    if len(sys.argv) != 5:
        sys.exit("cut_casia.py <casia_feats.z> <casia_labels.z> <n_imgs> <out_dir>")

    dataset_path = os.path.abspath(os.path.expanduser(sys.argv[1]))
    labels_path  = os.path.abspath(os.path.expanduser(sys.argv[2]))
    n_imgs       = int(sys.argv[3])
    out_dir      = os.path.abspath(os.path.expanduser(sys.argv[4]))

    validate_inputs(dataset_path, labels_path, n_imgs, out_dir)

    print("Loading the Dataset")
    X = np.memmap(dataset_path, dtype="float32", mode="r")
    print("Loading the Label set")
    y = np.fromfile(labels_path, dtype="int32")
    X = np.reshape(X, (y.size, X.size / y.size))

    t = time.time()
    print("\n- Cutting the dataset\n")
    idxs, unchosen_idxs = cut_dataset(y, n_imgs)

    pdb.set_trace()

    print("- Saving the new dataset")
    out_dataset  = os.path.join(out_dir, "cut_casia_feats.z")
    out_labelset = os.path.join(out_dir, "cut_casia_labels.z")
    save_sets(X, y, idxs, out_dataset, out_labelset)

    print("- Saving the unchosen dataset")
    out_dataset  = os.path.join(out_dir, "unchosen_casia_feats.z")
    out_labelset = os.path.join(out_dir, "unchosen_casia_labels.z")
    save_sets(X, y, unchosen_idxs, out_dataset, out_labelset)




    print("\nDone...")


if __name__ == "__main__":
    main()



