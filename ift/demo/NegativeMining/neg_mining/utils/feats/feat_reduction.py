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


def validate_inputs(method, dataset_path, n_rows, n_cols, n_bands, out_filename, pool_side, stride):
    valid_methods = {"sum", "hvsum", "pooling"}
    if method not in valid_methods:
        print_error("Invalid method \"%s\"... Try sum, hvsum or pooling!" % method, "validate_inputs")
        
    if not os.path.exists(dataset_path):
        print_error("Dataset \"%s\" does not exist!" % dataset_path, "validate_inputs")

    if n_rows <= 0:
        print_error("Invalid Number of Rows (ysize) %d... Try > 0" % n_rows, "validate_inputs")

    if n_cols <= 0:
        print_error("Invalid Number of cols (xsize) %d... Try > 0" % n_cols, "validate_inputs")

    if n_bands <= 0:
        print_error("Invalid Number of bands (zsize) %d... Try > 0" % n_bands, "validate_inputs")
 
    out_dir = os.path.dirname(out_filename)
    if os.path.exists(out_dir):
        if not os.path.isdir(out_dir):
            msg = "Out dir is a file and not a dir: %s" % out_dir
            print_error(msg, "validate_inputs")
    else:
        print("- Creating the output dir: %s\n" % out_dir)
        os.makedirs(out_dir)

    if method == "pooling":
        if stride <= 0:
            print_error("Invalid Stride: %d... Try > 0" % stride, "validate_inputs")

        if pool_side <= 0:
            print_error("Invalid Pooling Side: %d... Try > 0" % pool_side, "validate_inputs")
   






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



def max_per_band(mimg, **kwargs):
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


def sum_per_band(mimg, **kwargs):
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


def hvsum_per_band(mimg, **kwargs):
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



def pooling(mimg, **kwargs):
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
    pool_side = kwargs["pool_side"]
    stride    = kwargs["stride"]

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


def save_sets(X, out_dataset):
    """Save the new dataset.

    Parameters
    ----------
    X: numpy array with shape (n_samples, n_feats)
        The original dataset.

    out_dataset: string
        Output dataset pathname.

    """
    print("- Saving %s" % out_dataset)
    dataset_file = open(out_dataset, "wb")
    X.tofile(dataset_file)
    dataset_file.close()




def feat_reduction(method, X, n_rows, n_cols, n_bands, pool_side, stride):
    """Reduce the feature space from the dataset X using the method <method>.
    """
    # assign the correct reduction method, according to the parameter <method>
    reduc_function = None
    if method == "sum":
        reduc_function = sum_per_band
        kargs = {}
    elif method == "hvsum":
        reduc_function = hvsum_per_band
        kargs = {}
    elif method == "pooling":
        reduc_function = pooling
        kargs = {"pool_side": pool_side, "stride": stride}
    else:
        print_error("Invalid method \"%s\"... Try sum, hvsum or pooling!" % method, "feat_reduction")
    
    X_reduc   = []
    for s, feat_vector in enumerate(X):
        print("[%d/%d]" % (s+1, X.shape[0]))
        mimg = feat_2_mimg(feat_vector, n_rows, n_cols, n_bands)

        # feature space reduction
        X_reduc.append(reduc_function(mimg, **kargs))
    X_reduc   = np.array(X_reduc)

    return X_reduc





def main():
    """Reduce the feature space by Sum, Max pooling, hvSum, and Pooling.
    """
    if (sys.argv[1].lower() != "pooling") and (len(sys.argv) != 7):
        sys.exit("reduce_feats.py <method: sum | hvsum | pooling> <dataset.z> <n_rows> <n_cols> <n_bands> <out_filename> [<pool_side> <stride>]")
    elif (sys.argv[1].lower() == "pooling") and (len(sys.argv) != 9):
        msg = "reduce_feats.py <method: sum | hvsum | pooling> <dataset.z> <n_rows> <n_cols> <n_bands> <out_filename> [<pool_side> <stride>]\n"
        msg += "*** ps: pooling require the parameters <pool_size> and <stride>"
        sys.exit(msg)

    method       = sys.argv[1].lower()
    dataset_path = os.path.abspath(os.path.expanduser(sys.argv[2]))
    n_rows       = int(sys.argv[3])
    n_cols       = int(sys.argv[4])
    n_bands      = int(sys.argv[5])
    out_filename = os.path.abspath(os.path.expanduser(sys.argv[6]))

    if method == "pooling":
        pool_side    = int(sys.argv[7])
        stride       = int(sys.argv[8])
    else:
        pool_side = stride = None

    validate_inputs(method, dataset_path, n_rows, n_cols, n_bands, out_filename, pool_side, stride)

    t = time.time()

    print("Loading the Dataset")
    X = np.memmap(dataset_path, dtype="float32", mode="r")
    n_feats   = n_rows*n_cols*n_bands
    n_samples = X.size/n_feats
    X.resize(n_samples, n_feats)

    print("- Generating the reduced datasets")
    X_reduc = feat_reduction(method, X, n_rows, n_cols, n_bands, pool_side, stride)

    save_sets(X_reduc, out_filename)

    print("\nTime elapsed: %f secs" % (time.time()-t))



if __name__ == "__main__":
    main()



