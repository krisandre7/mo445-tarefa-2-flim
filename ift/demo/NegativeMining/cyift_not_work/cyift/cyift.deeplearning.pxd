from cyift.dataset cimport *

cdef extern from "../libift/include/iftDeepLearning.h":
    cdef iftDataSet *iftCNNFeatReductionByOPF(iftDataSet *Z, int nfeats, int total_of_patches, int num_rep_patches, float kmax)