cdef extern from "../../../../include/iftMatrix.h":
    ctypedef struct iftMatrix:
        double *val
        int ncols
        int nrows
        int *tbrow
        int n

cdef extern from "../../../../include/iftDataSet.h":
    ctypedef float (*iftArcWeightFun)(float *f1, float *f2, float *alpha, int n)
    
    ctypedef struct iftSample:
        float *feat
        int truelabel
        int label
        int id
        float weight
        unsigned char status
    
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
        
    cdef iftDataSet *iftReadOPFDataSet(char *filename)
    cdef void iftDestroyDataSet(iftDataSet **Z)