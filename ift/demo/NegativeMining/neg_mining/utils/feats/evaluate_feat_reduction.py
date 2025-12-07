import numpy as np
import os
import pdb
import sys
import time

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

def print_error(error_msg, function):
    sys.exit("\nERROR in function {0}\n{1}".format(function, error_msg))


def validate_inputs(dataset_path, labels_path, n_rows, n_cols, n_bands, pool_side, stride, out_dir):
    if not os.path.exists(dataset_path):
        print_error("Dataset \"%s\" does not exist!" % dataset_path, "validate_inputs")

    if not os.path.exists(labels_path):
        print_error("Labels \"%s\" does not exist!" % labels_path, "validate_inputs")

    if n_rows <= 0:
        print_error("Invalid Number of Rows (ysize) %d... Try > 0" % n_rows, "validate_inputs")

    if n_cols <= 0:
        print_error("Invalid Number of cols (xsize) %d... Try > 0" % n_cols, "validate_inputs")

    if n_bands <= 0:
        print_error("Invalid Number of bands (zsize) %d... Try > 0" % n_bands, "validate_inputs")

    if stride <= 0:
        print_error("Invalid Stride: %d... Try > 0" % stride, "validate_inputs")

    if pool_side <= 0:
        print_error("Invalid Pooling Side: %d... Try > 0" % pool_side, "validate_inputs")

    if os.path.exists(out_dir):
        if not os.path.isdir(out_dir):
            msg = "Out dir is a file and not a dir: %s" % out_dir
            print_error(msg, "validate_inputs")
    else:
        print("- Creating the output dir: %s\n" % out_dir)
        os.makedirs(out_dir)






def save_sets(X, y, idxs, out_dir):
    """Save the new dataset.

    Parameters
    ----------
    X: numpy array with shape (n_samples, n_feats)
        The original dataset.

    y: numpy array with shape (n_samples,)
        The label set.

    idxs: numpy array
        The indices from X and y of the chosen samples.

    out_dir: string
        Output directory.
    """
    dataset_file = open(os.path.join(out_dir, "cut_pubfig83_feats.z"), "wb")
    label_file   = open(os.path.join(out_dir, "cut_pubfig83_labels.z"), "wb")
    X_out = X[idxs]
    y_out = y[idxs]

    X_out.tofile(dataset_file)
    y_out.tofile(label_file)

    dataset_file.close()
    label_file.close()



def feat_2_mimg(feat_vector, n_rows, n_cols, n_bands):
    """ Transform a 1D feature vector of size n_bands*n_rows*n_cols into a multi-band image of (n_bands,n_rows,n_cols)
    """
    n_feats = n_rows*n_cols*n_bands

    mimg = np.zeros((n_bands,n_rows,n_cols))
    for i in xrange(n_rows):
        for j in xrange(n_cols):
            p = ((i*n_cols)+j) * n_bands
            q = p+n_bands
            # the python way is: [band, row, col]
            mimg[:,i,j] = feat_vector[p:q]

    return mimg



def max_per_band(mimg):
    """Build a feature vector with the maximum value of each band from the multi-band image.

    Parameters
    ----------
    mimg: 3D numpy array of shape (n_bands, n_rows, n_cols)
        Multi-band to be pooled.

    Returns
    -------
    feat_vector: 1D numpy array of shape (n_bands,).
        1D Numpy array which represents the feature vector.
    """
    n_bands, n_rows, n_cols = mimg.shape

    feat_vect = []

    for b in xrange(n_bands):
        feat_vect.append(np.amax(mimg[b]))

    feat_vect = np.array(feat_vect, dtype="float32")
    return feat_vect


def sum_per_band(mimg):
    """Build a feature vector with the sum value of each band from the multi-band image.

    Parameters
    ----------
    mimg: 3D numpy array of shape (n_bands, n_rows, n_cols)
        Multi-band to be pooled.

    Returns
    -------
    feat_vector: 1D numpy array of shape (n_bands,).
        1D Numpy array which represents the feature vector.
    """
    n_bands, n_rows, n_cols = mimg.shape

    feat_vect = []

    for b in xrange(n_bands):
        feat_vect.append(np.sum(mimg[b]))

    feat_vect = np.array(feat_vect, dtype="float32")
    return feat_vect 


def hvsum_per_band(mimg):
    """Build a feature vector with the sum of each row and col of each band from the multi-band image.

    Parameters
    ----------
    mimg: 3D numpy array of shape (n_bands, n_rows, n_cols)
        Multi-band to be pooled.

    Returns
    -------
    feat_vector: 1D numpy array of shape ((n_rows+n_cols)*n_bands,).
        1D Numpy array which represents the feature vector.
    """
    n_bands, n_rows, n_cols = mimg.shape

    feat_vect = []

    for b in xrange(n_bands):
        feat_vect.extend(np.sum(mimg[b], axis=0)) # sum the columns
        feat_vect.extend(np.sum(mimg[b], axis=1)) # sum the rows

    feat_vect = np.array(feat_vect, dtype="float32")
    return feat_vect



def pooling(mimg, pool_side, stride):
    """ Applies a Pooling into a multi-band image, returning a 1D feature vector.
    Compute the average value of a pooling window.
    Pooling windows is a square of side pool_side.

    Parameters
    ----------
    mimg: 3D numpy array of shape (n_bands, n_rows, n_cols)
        Multi-band to be pooled.

    pool_side: int
        Pooling window side.

    stride: int
        Stride used in the pooling.

    Returns
    -------
    feat_vector: 1D numpy array of shape (n_feats_f,).
        1D Numpy array which represents the feature vector.
        n_feats_f = (n_bands_f, n_rows_f, n_cols_f)
        
        n_bands_f = n_bands
        n_rows_f  = ceil((n_rows-pool_side+1)/stride)
        n_cols_f  = ceil((n_cols-pool_side+1)/stride)
    """
    n_bands, n_rows, n_cols = mimg.shape
    pool_window = pool_side*pool_side*1.0

    feat_vect = []

    for b in xrange(n_bands):
        for i in xrange(0,n_rows,stride):
            for j in xrange(0,n_cols,stride):
                avg = np.sum(mimg[b, i:(i+pool_side), j:(j+pool_side)])
                avg /= pool_window
                feat_vect.append(avg)

    feat_vect = np.array(feat_vect, dtype="float32")
    return feat_vect



def feat_reduction(X, n_rows, n_cols, n_bands, pool_side, stride):
    """Reduce the feature space from the dataset X using 4 methods: Max, Sum, and hvSum.
    """
    X_max   = []
    X_sum   = []
    X_hvsum = []
    X_pool  = []
    X_pool2 = []
    for s, feat_vector in enumerate(X):
        print("[%d/%d]" % (s+1, X.shape[0]))
        mimg = feat_2_mimg(feat_vector, n_rows, n_cols, n_bands)

        # feature space reductions
        X_max.append(max_per_band(mimg))
        X_sum.append(sum_per_band(mimg))
        X_hvsum.append(hvsum_per_band(mimg))
        X_pool.append(pooling(mimg, pool_side, stride))
        X_pool2.append(pooling(mimg, 5, 5))
    X_max   = np.array(X_max)
    X_sum   = np.array(X_sum)
    X_hvsum = np.array(X_hvsum)
    X_pool  = np.array(X_pool)
    X_pool2 = np.array(X_pool2)

    return X_max, X_sum, X_hvsum, X_pool, X_pool2


def main():
    """Evaluate the accuracy of the feature space reduction strategies: Sum, Max pooling, hvSum, and Pooling.
    """
    if len(sys.argv) != 9:
        sys.exit("evaluate_feat_reduction.py <dataset.z> <labels_set.z> <n_rows> <n_cols> <n_bands> <pool_side> <stride> <out_dir>")

    dataset_path = os.path.abspath(os.path.expanduser(sys.argv[1]))
    labels_path  = os.path.abspath(os.path.expanduser(sys.argv[2]))
    n_rows       = int(sys.argv[3])
    n_cols       = int(sys.argv[4])
    n_bands      = int(sys.argv[5])
    pool_side    = int(sys.argv[6])
    stride       = int(sys.argv[7])
    out_dir      = os.path.abspath(os.path.expanduser(sys.argv[8]))

    validate_inputs(dataset_path, labels_path, n_rows, n_cols, n_bands, pool_side, stride, out_dir)

    t = time.time()

    print("Loading the Dataset")
    X = np.fromfile(dataset_path, dtype="float32")
    y = np.fromfile(labels_path, dtype="int32")
    n_feats   = n_rows*n_cols*n_bands
    n_samples = X.size/n_feats
    X.resize(n_samples, n_feats)

    if X.shape[0] != y.size:
        print_error("Number of samples is inconsistent between X and y: %d - %d" % (X.shape[0], y.size), "main")

    X_max, X_sum, X_hvsum, X_pool, X_pool2 = feat_reduction(X, n_rows, n_cols, n_bands, pool_side, stride)

    # generating a set of training and testing idxs for 10 iterations
    sss = StratifiedShuffleSplit(y, 10, train_size=0.9)

    orig_accs  = []
    max_accs   = []
    sum_accs   = []
    hvsum_accs = []
    pool_accs  = []
    pool2_accs = []

    for train_idxs, test_idxs in sss:
        clf_orig  = OneVsRestClassifier(svm.SVC(C=1e5, kernel="precomputed"))
        clf_max   = OneVsRestClassifier(svm.SVC(C=1e5, kernel="precomputed"))
        clf_sum   = OneVsRestClassifier(svm.SVC(C=1e5, kernel="precomputed"))
        clf_hvsum = OneVsRestClassifier(svm.SVC(C=1e5, kernel="precomputed"))
        clf_pool  = OneVsRestClassifier(svm.SVC(C=1e5, kernel="precomputed"))
        clf_pool2 = OneVsRestClassifier(svm.SVC(C=1e5, kernel="precomputed"))

        X_train       = X[train_idxs]
        X_max_train   = X_max[train_idxs]
        X_sum_train   = X_sum[train_idxs]
        X_hvsum_train = X_hvsum[train_idxs]
        X_pool_train  = X_pool[train_idxs]
        X_pool2_train = X_pool2[train_idxs]
        y_train       = y[train_idxs]

        X_test       = X[test_idxs]
        X_max_test   = X_max[test_idxs]
        X_sum_test   = X_sum[test_idxs]
        X_hvsum_test = X_hvsum[test_idxs]
        X_pool_test  = X_pool[test_idxs]
        X_pool2_test = X_pool2[test_idxs]
        y_test       = y[test_idxs]

        print("- Building the Training Kernel")
        kernel_train = np.dot(X_train, X_train.T)
        kernel_max_train = np.dot(X_max_train, X_max_train.T)
        kernel_sum_train = np.dot(X_sum_train, X_sum_train.T)
        kernel_hvsum_train = np.dot(X_hvsum_train, X_hvsum_train.T)
        kernel_pool_train = np.dot(X_pool_train, X_pool_train.T)
        kernel_pool2_train = np.dot(X_pool2_train, X_pool2_train.T)

        print("- SVM Training")
        clf_orig.fit(kernel_train, y_train)
        del kernel_train
        clf_max.fit(kernel_max_train, y_train)
        del kernel_max_train
        clf_sum.fit(kernel_sum_train, y_train)
        del kernel_sum_train
        clf_hvsum.fit(kernel_hvsum_train, y_train)
        del kernel_hvsum_train
        clf_pool.fit(kernel_pool_train, y_train)
        del kernel_pool_train
        clf_pool2.fit(kernel_pool2_train, y_train)
        del kernel_pool2_train

        print("- SVM Predicting")
        y_pred       = clf_orig.predict(np.dot(X_test, X_train.T))
        y_max_pred   = clf_max.predict(np.dot(X_max_test, X_max_train.T))
        y_sum_pred   = clf_sum.predict(np.dot(X_sum_test, X_sum_train.T))
        y_hvsum_pred = clf_hvsum.predict(np.dot(X_hvsum_test, X_hvsum_train.T))
        y_pool_pred = clf_pool.predict(np.dot(X_pool_test, X_pool_train.T))
        y_pool2_pred = clf_pool2.predict(np.dot(X_pool2_test, X_pool2_train.T))

        print("- Computing the Score\n")
        orig_accs.append(accuracy_score(y_test, y_pred))
        max_accs.append(accuracy_score(y_test, y_max_pred))
        sum_accs.append(accuracy_score(y_test, y_sum_pred))
        hvsum_accs.append(accuracy_score(y_test, y_hvsum_pred))
        pool_accs.append(accuracy_score(y_test, y_pool_pred))
        pool2_accs.append(accuracy_score(y_test, y_pool2_pred))

    orig_accs  = np.array(orig_accs)
    max_accs   = np.array(max_accs)
    sum_accs   = np.array(sum_accs)
    hvsum_accs = np.array(hvsum_accs)
    pool_accs  = np.array(pool_accs)
    pool2_accs  = np.array(pool2_accs)

    print("- acc = %.4f +- %.4f" % (np.mean(orig_accs), np.std(orig_accs)))
    print("- acc_max = %.4f +- %.4f" % (np.mean(max_accs), np.std(max_accs)))
    print("- acc_sum = %.4f +- %.4f" % (np.mean(sum_accs), np.std(sum_accs)))
    print("- acc_hvsum = %.4f +- %.4f" % (np.mean(hvsum_accs), np.std(hvsum_accs)))
    print("- acc_pool = %.4f +- %.4f" % (np.mean(pool_accs), np.std(pool_accs)))
    print("- acc2_pool = %.4f +- %.4f\n" % (np.mean(pool2_accs), np.std(pool2_accs)))

    print("\nTime elapsed: %f secs" % (time.time()-t))


if __name__ == "__main__":
    main()



