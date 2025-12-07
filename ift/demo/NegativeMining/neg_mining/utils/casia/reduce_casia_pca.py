import numpy as np
import os
import pdb
from sklearn.decomposition import PCA
import sys
import time




def print_error(msg, function):
    sys.exit("ERROR in %s: %s" % (function, msg))




def validate_inputs(dataset_path, labels_path, n_img_train, out_filename):
    if not os.path.exists(dataset_path):
        print_error("Dataset \"%s\" does not exist!" % dataset_path, "validate_inputs")

    if not os.path.exists(labels_path):
        print_error("Labels \"%s\" does not exist!" % labels_path, "validate_inputs")

    if n_img_train <= 0:
        print_error("Invalid Number of Images to Train a PCA %d... Try > 0" % n_img_train, "validate_inputs")

    out_dir = os.path.dirname(out_filename)

    if os.path.exists(out_dir):
        if not os.path.isdir(out_dir):
            msg = "Out dir is a file and not a dir: %s" % out_dir
            print_error(msg, "validate_inputs")
    else:
        print("- Creating the output dir: %s\n" % out_dir)
        os.makedirs(out_dir)




def get_neg_indices(y, n_min_imgs):
    """ Get a set with the indices of the negative images from the base.
    Since that CASIA has a large range of images per individuals, we are considering
    that the images from the individuals which does not have at least <n_min_imgs>
    are sure to be negatives.

    Parameters
    ----------
    y: numpy array
        Numpy Array with the labels of the images

    n_min_imgs: int
        Minimum number of images that a non-negative individual must have.

    Returns
    -------
    neg_indices: numpy array
        Numpy Array with the indices of the negative images.    

    """
    neg_indices = []

    for label in np.unique(y):
        img_idxs = np.where(y==label)[0]
        # print("Label: %d - %d imgs" % (label, img_idxs.size))

        if img_idxs.size < 50:
            neg_indices.extend(img_idxs)


    return np.array(neg_indices)



def apply_PCA(proj, X, reduc_dataset_file):
    n_samples, n_feats = X.shape

    print("- Projecting Positive Testing Samples on subspace")
    t = time.time()

    count = 0
    # processes 100 samples per iteration
    for idx in xrange(0, n_samples, 100):
        print("[%d/%d]" % (idx, n_samples))
        X_train = X[idx:(idx+100)]
	count += X_train.shape[0]
	X_reduc = proj.transform(X[idx:(idx+100)])
        X_reduc = X_reduc.flatten() # transforms the array to 1D
        X_reduc.tofile(reduc_dataset_file)

    print("* count = %d, X.shape[0] = %d" % (count, X.shape[0]))
    print("--> Time elapsed to transform the entire dataset: {0:.2f} secs\n".format(time.time()-t))



def main():
    if len(sys.argv) != 5:
        sys.exit("reduce_casia_pca.py <casia_feats.z> <casia_labels.z> <n_img_train_PCA> <out_filename>")

    t_total = time.time()

    dataset_path = os.path.abspath(os.path.expanduser(sys.argv[1]))
    labels_path  = os.path.abspath(os.path.expanduser(sys.argv[2]))
    n_img_train  = int(sys.argv[3])
    out_filename = os.path.abspath(os.path.expanduser(sys.argv[4]))

    validate_inputs(dataset_path, labels_path, n_img_train, out_filename)

    print(dataset_path)
    print(labels_path)
    print(n_img_train)
    print(out_filename)


    X = np.memmap(dataset_path, dtype="float32", mode="r")
    y = np.fromfile(labels_path, dtype="int32")
    X.resize(y.size, X.size / y.size)

    t = time.time()
    print("\n- Getting the indices from the negative images\n")
    neg_indices       = get_neg_indices(y, 50)
    train_neg_indices = np.random.choice(neg_indices, n_img_train, replace=False)
    X_train           = X[train_neg_indices]
    n_comps           = min(X_train.shape)
    print("--> Time elapsed to Get the indice set and to assign the X_train: {0:.2f} secs\n".format(time.time()-t))

    print("- Learning PCA subspace")
    t_fit = time.time()
    proj = PCA(n_components=n_img_train)
    proj.fit(X=X_train)
    del X_train
    print("--> Time elapsed to Fit the model: {0:.2f} secs\n".format(time.time()-t_fit))


    reduc_dataset_file = open(out_filename, "wb")
    apply_PCA(proj, X, reduc_dataset_file)
    reduc_dataset_file.close()

    print("****** TOTTAL TIME: {0:.2f} secs\n".format(time.time()-t_total))



if __name__ == "__main__":
    main()






