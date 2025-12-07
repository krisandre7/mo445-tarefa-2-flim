from cyift.common cimport *
from cyift.matrix cimport *

cdef extern from "../libift/include/iftDataSet.h":
    cdef int TEST
    cdef int IFT_TRAIN
    cdef int IFT_OUTLIER
    cdef int ERROR
    
    cdef int WEIGHT
    cdef int LABEL
    cdef int CLASS
    cdef int POINT
    cdef int STATUS
    
    ctypedef float (*iftArcWeightFun)(float *f1, float *f2, float *alpha, int n)
    
    ctypedef struct iftSample:
        float *feat
        int truelabel
        int label
        int id
        float weight
        uchar status
    
    ctypedef struct iftFeatSpaceParam:
        float *mean
        float *stdev
        int nfeats
        char *w
        int ncomps
        iftMatrix *R
        iftMatrix *W
    
    ctypedef struct iftDataSet:
        iftSample *sample
        int nfeats
        int nsamples
        int nlabels
        int nclasses
        int ntrainsamples
        iftArcWeightFun iftArcWeight
        float *alpha
        void *ref_data
        iftFeatSpaceParam fsp
        
    cdef iftDataSet *iftCreateDataSet(int nsamples, int nfeats)
    cdef iftDataSet *iftCreateDataSetWithoutFeatsVector(int nsamples, int nfeats)
    cdef iftDataSet *iftReadOPFDataSet(char *filename)
    cdef void iftWriteOPFDataSet(iftDataSet *Z, char *filename)
    cdef void iftDestroyDataSet(iftDataSet **Z)
    cdef void iftDestroyDataSet2(iftDataSet **Z)
    cdef iftDataSet *iftCopyDataSet(iftDataSet *Z)
    cdef iftDataSet *iftCopyDataSet2(iftDataSet *Z)
    
    cdef void iftSetStatus(iftDataSet *Z, uchar status)
    cdef iftDataSet *iftNormalizeDataSetByZScore(iftDataSet *Ztrain)
    cdef iftDataSet *iftNormalizeTestDataSet(iftDataSet *Ztrain, iftDataSet *Ztest)
    cdef iftDataSet *iftNormOneDataSet(iftDataSet *Z)
    
    cdef iftDataSet *iftCentralizeDataSet(iftDataSet *Z)
    cdef iftDataSet *iftTransFeatSpaceByPCA(iftDataSet *Z, int num_of_comps)
    cdef iftDataSet *iftCentralizeTestDataSet(iftDataSet *Ztrain, iftDataSet *Ztest)
    cdef iftDataSet *iftTransformTestDataSetByPCA(iftDataSet *Ztrain, iftDataSet *Ztest)
    
    cdef iftDataSet *iftBuildPatchDataSet(iftDataSet *Z, int patch, int nfeats)
    cdef iftDataSet *iftBuildPatchDataSetFromSample(iftSample sample, int num_of_patches, int nfeats)
    
    cdef void iftSetDistanceFunction(iftDataSet *Z, int function_number)


# Customized functions
cdef iftDataSet* create_dataset(int n_samples, int n_feats, int n_classes=?, alloc_feats=?)
cdef void destroy_dataset(iftDataSet **Z, dealloc_feats=?)
cdef iftDataSet* copy_dataset(iftDataSet *Z, copy_feats=?)
cdef iftDataSet* build_dataset_from_idxs(idxs, iftDataSet* Zorig, copy_feats=?)
cdef iftDataSet* build_dataset_from_feats(sample_filenames, true_labels)

cdef void truelabels_to_value(iftDataSet **Z, int truelabel)
cdef void set_truelabel_as_pos_into_dataSet(iftDataSet **Z, int pos_truelabel)

cdef void assign_label_from_max_scores(iftDataSet **Z, score_matrix, labels)
cdef void print_label_density(iftDataSet *Z, log=?)
    
cdef dataset_to_numpy(iftDataSet *Z)
cdef iftDataSet* numpy_to_dataset(dataset, n_classes=?)

cdef get_orig_ids(iftDataSet *Z, idxs)
    
    
    
    
    
    
    
    