from libc.stdlib cimport malloc, free, calloc
from libc.stdio cimport printf
import numpy as np

from cyift.dataset cimport *
from cyift.descriptors cimport *
import common


cdef iftDataSet* create_dataset(int n_samples, int n_feats, int n_classes=0, alloc_feats=True):
    """Create an iftDataSet.
    
    Parameters
    ----------
    n_samples: int
        Number of Samples of the iftDataSet.
    
    n_feats: int
        Number of Features of the iftDataSet.
    
    n_classes: int, default 0
        Number of Classes of the iftDataSet.
    
    alloc_feats: boolean, default True
        If this flag is True, the feature vectors are allocated. Otherwise, it is only created the pointers.
        
    Returns
    -------
    Z: iftDataSet*
        An iftDataSet*.
    """
    cdef iftDataSet *Z = NULL
    if (alloc_feats):
        Z = iftCreateDataSet(n_samples, n_feats)
    else:
        Z = iftCreateDataSetWithoutFeatsVector(n_samples, n_feats)
    Z.nclasses = n_classes

    return Z


cdef void destroy_dataset(iftDataSet **Z, dealloc_feats=True):
    """Destroy an iftDataSet.
    
    Parameters:
    Z: iftDataSet**
        Address of the iftDataSet* to be destroyed/deallocated.
    
    dealloc_feats: boolean, default True
        If True, the feature vectors are destroyed.
    """
    cdef iftDataSet *Zaux = Z[0] # Z[0] in Cython is equivalent to *Z in C
    
    if (dealloc_feats):
        iftDestroyDataSet(&Zaux)
    else:
        iftDestroyDataSet2(&Zaux)
    
    Zaux = NULL


cdef iftDataSet* copy_dataset(iftDataSet *Z, copy_feats=True):
    """Copy/Clone an iftDataSet.
    
    Parameters
    ----------
    Z: iftDataSet*
        The iftDataSet to be copied.
    
    copy_feats: boolean
        If copy_feats = True, the feature vectors from Z will be copied. Otherwise, they will only be referenced. 
    
    Returns
    -------
    Zout: iftDataSet*
        The copied iftDataSet.
    """
    if copy_feats:
        return iftCopyDataSet(Z)
    else:
        return iftCopyDataSet2(Z)


cdef iftDataSet* build_dataset_from_idxs(idxs, iftDataSet* Zorig, copy_feats=False):
    """Build an iftDataSet from a list of samples indices of another iftDataSet.
    
    Parameters
    ----------
    idxs: list of int
        List of sample indices from Zorig that will be copied.
    
    Zorig: iftDataSet*
        Original iftDataSet used to copy samples from a sample index list.
    
    copy_feats: boolean, default False
        If True, the function allocates the feature vectors in the Output iftDataSet and copy all 
        features of each sample from Zorig.
        Otherwise, just point the feature vectors from Zorig to Zout.
    
    Returns
    -------
    Zout: iftDataSet*
        Output iftDataSet built from idxs and Zorig.
    """
    if copy_feats:
        alloc_feats_=True
    else:
        alloc_feats_=False
    
    cdef iftDataSet *Zout = create_dataset(len(idxs), Zorig.nfeats, n_classes=0, alloc_feats=alloc_feats_)
    true_labels = []
    
    cdef int s = 0
    for idx in idxs:
        Zout.sample[s].id = Zorig.sample[idx].id;
        Zout.sample[s].truelabel = Zorig.sample[idx].truelabel;
        true_labels.append(Zout.sample[s].truelabel)
        Zout.sample[s].label = Zorig.sample[idx].label;
        Zout.sample[s].status = Zorig.sample[idx].status;
        Zout.sample[s].weight = Zorig.sample[idx].weight;
        if copy_feats:
            for f in xrange(Zorig.nfeats):
                Zout.sample[s].feat[f] = Zorig.sample[idx].feat[f]
        else:
            Zout.sample[s].feat = Zorig.sample[idx].feat # copy the feature vector address
        s += 1
    
    Zout.nclasses = len(set(true_labels))
    
    return Zout


cdef iftDataSet* build_dataset_from_feats(sample_filenames, true_labels):
    """Build a DataSet from the feature vectors and their true labels.
    
    Parameters
    ----------
    sample_filenames: list of string with the shape (n_samples,)
        The index i of the list corresponds to the filename of the feature vector from the sample i.
    
    true_labels: list of int with the shape (n_samples,)
        The index i of the list corresponds to the true label from the sample i.
    
    Notes
    -----
    E.g: let sample_filenames[10] = "base/00001_000010.feats", true_label[10] = 1.
    Then, the sample 10 has filename "base/00001_000010.feats" and true label = 1.
    
    When building the iftDataSet, the function assigns the sample id as the sample idx.
    E.g: the sample 10 will have: Z.sample[10].id = 10.
    """
    cdef int nsamples = len(sample_filenames)
    cdef int nclasses = len(set(true_labels))
    
    if (nsamples <= 0):
        common.print_error("Invalid Number of Samples: %d... Try > 0" % nsamples, "cyift.dataset.build_dataset_from_feats")
    if (nclasses <= 0):
        common.print_error("Invalid Number of Classes: %d... Try > 0" % nclasses, "cyift.dataset.build_dataset_from_feats")
    
    print("- Number of Images/Samples: %d" % nsamples);
    print("- Number of Classes: %d" % nclasses);
    
    
    cdef iftFeatures *feat = NULL
    feat = iftReadFeatures2(sample_filenames[0])
    cdef iftDataSet *Z = create_dataset(nsamples, feat.n, nclasses, alloc_feats=False)
    iftDestroyFeatures(&feat)
    
#     cdef iftFeatures **feats = <iftFeatures**> calloc (Z.nsamples, sizeof (iftFeatures*))
    print("- Building Datasets from Feats\n")
#     
    for i in xrange(nsamples):
        feat = iftReadFeatures2(sample_filenames[i])
        Z.sample[i].feat = feat.val
        Z.sample[i].truelabel = true_labels[i]
        Z.sample[i].id = i
    
    return Z


cdef void truelabels_to_value(iftDataSet **Z, int truelabel):
    """Set the truelabel of all samples to a given truelabel value.

    Parameters
    ----------
    Z: iftDataSet**
        Reference to an iftDataSet to have the assigned labels.

    truelabel: int
        Label that will be setted in all samples from Z.

    Returns
    -----
    There is not an explicit return, however the labels are assigned in the *Z.
    """
    cdef iftDataSet *Zaux = Z[0] # Z[0] in Cython is equivalent to *Z in C
    
    for s in xrange(Zaux.nsamples):
        Zaux.sample[s].truelabel = truelabel
    Zaux.nclasses = 1
    

cdef void set_truelabel_as_pos_into_dataSet(iftDataSet **Z, int pos_truelabel):
    """Set the sample truelabel as 1 if the sample.truelabel = pos_truelabel.
    Otherwise, set sample.truelabel = 2 (negative sample).

    Parameters
    ----------
    Z: iftDataSet**
        Reference to an iftDataSet to have the assigned labels.

    pos_truelabel: int
        Positive label used to assign the new label 1 or 2 in Z.

    Returns
    -----
    There is not an explicit return, however the labels are assigned in the *Z.
    """    
    cdef int neg_flag = 0
    cdef int pos_flag = 0
    cdef iftDataSet *Zaux = Z[0] # Z[0] in Cython is equivalent to *Z in C

    for s in xrange(Zaux.nsamples):
        if (Zaux.sample[s].truelabel == pos_truelabel):
            Zaux.sample[s].truelabel = 1
            pos_flag = 1
        else:
            Zaux.sample[s].truelabel = 2
            neg_flag = 1
           
    Zaux.nclasses = neg_flag + pos_flag


cdef void assign_label_from_max_scores(iftDataSet **Z, score_matrix, labels):
    """Assign the labels getting the max scores.

    Parameters
    ----------
    Z: iftDataSet** with shape (n_samples, n_feats)
        Reference to the target iftDataSet* to have the assigned labels.

    score_matrix: NumPy array with shape (n_samples, n_labels)
        Score Matrix that will assign the labels from the max scores for each sample.

    labels: list of int with shape (n_labels)
        List with the labels to be assigned.

    Notes
    -----
    Each column i of the score_matrix corresponds to the learned hyperplane from the labels[i].
    """
    cdef iftDataSet *Zaux = Z[0] # Z[0] in Cython is equivalent to *Z in C
    
    if (Zaux.nsamples != len(score_matrix)):
        common.print_error("Number of Samples is different between Dataset and Score Matrix: (%d, %d)"
                           % (Zaux.nsamples, len(score_matrix)), "cyift.dataset.assign_label_from_max_scores")

    if (score_matrix.shape[1] != len(labels)):
        common.print_error("Number of Learned Hyperplanes (labels) is different Number of True Labels: (%d, %d)"
                           % (score_matrix.shape[1], len(labels)), "cyift.dataset.assign_label_from_max_scores")
    
    assigned_labels = []
    for s, scores in enumerate(score_matrix):
        # hyperplane_idx = np.argmax(scores) gives the idx from the hyperplane with higher score for the sample s.
        # This hyperplane belongs to the label -> labels[hyperplane_idx]
        hyperplane_idx = np.argmax(scores)
        Zaux.sample[s].label = labels[hyperplane_idx]
        assigned_labels.append(Zaux.sample[s].label)
    Zaux.nlabels = len(set(assigned_labels))
    


cdef void print_label_density(iftDataSet *Z, log=None):
    """Print the amount of samples in each available label.

    Parameters
    ----------
    Z: iftDataSet*
        Dataset to be analyzed.

    log: file
        Log file.
    """
    cluster_density = [0] * Z.nlabels

    for s in xrange(Z.nsamples):
        cluster_density[Z.sample[s].label-1] += 1 # idx i in cluster_density corresponds to label i+1 
 
    if log:
        log.write("--- Cluster Density ---\n")
        for label in xrange(Z.nlabels):
            log.write("- Cluster: %d - %d samples\n" % (label+1, cluster_density[label]))
        log.write("\n")
    else:
        print("--- Cluster Density ---")
        for label in xrange(Z.nlabels):
            print("- Cluster: %d - %d samples" % (label+1, cluster_density[label]))
        print("")


cdef dataset_to_numpy(iftDataSet *Z):
    """Convert an iftDataSet* into a NumPy Array.

    Parameters
    ----------
    Z: iftDataSet*
        The iftDataSet to be converted.

    Returns
    -------
    dataset: NumPy array with the shape (Z.nsamples, Z.nfeats)
        Coverted dataset.

    Notes
    -----
    The number of classes from Z is not stored in the NumPy array.
    """
    dataset = np.zeros((Z.nsamples, Z.nfeats))
    
    for s in xrange(Z.nsamples):
        for f in xrange(Z.nfeats):
            dataset[s,f] = Z.sample[s].feat[f]
    
    return dataset


cdef iftDataSet* numpy_to_dataset(dataset, n_classes=0):
    """Convert a NumPy array into an iftDataSet*.

    Parameters
    ----------
    dataset: NumPy array with the shape (n_samples, n_feats)
        DataSet to be converted.

    n_classes: int, optional, default 0
        Number of Classes from the dataset.

    Returns
    -------
    Z: iftDataSet*
        The converted iftDataSet.    
    """

    cdef iftDataSet *Z = iftCreateDataSet(dataset.shape[0], dataset.shape[1])
    Z.nclasses = n_classes
    
    for s in xrange(Z.nsamples):
        for f in xrange(Z.nfeats):
            Z.sample[s].feat[f] = dataset[s,f]
    
    return Z



cdef get_orig_ids(iftDataSet *Z, idxs):
    """Get the ids of the samples.

    Parameters
    ----------
    Z: iftDataSet*
        Dataset to be used.

    idxs: list of int with shape (n_samples,)
        List of sample indices from Z. Values in [0..n_samples-1]

    Returns
    -------
    ids: list of int with the shape (n_samples,).
        List of the ids of the samples from idxs
    """
    return [Z.sample[s].id for s in idxs]