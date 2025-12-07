from cyift.libsvm cimport *
from cyift.dataset cimport * 

cdef extern from "../libift/include/iftSVM.h":
    ctypedef svm_node svmNode
    ctypedef svm_problem svmProblem
    ctypedef svm_parameter svmParameter
    ctypedef svm_model svmModel
    
    ctypedef struct iftSVM:
        svmParameter *params
        svmProblem *problem
        svmModel **model
        double *truelabel
        int nmodels
    
    ctypedef struct iftSVMHyperplane:
        int n # size of feature vector
        float *feat # feature values
        float bias


    cdef iftSVMHyperplane *iftCreateSVMHyperplane(int n)
    cdef void iftDestroySVMHyperplane(iftSVMHyperplane **h)
    cdef iftSVMHyperplane *iftSVMGetNormalHyperplane(iftSVM* svm, int idxModel, iftDataSet* Ztrain)
    cdef iftSVMHyperplane **iftSVMGetAllNormalHyperplanes(iftSVM *svm, iftDataSet *Ztrain) 


    cdef iftSVM *iftCreateLinearSVC(double C)
    cdef iftSVM *iftCreateRBFSVC(double C, double sigma)
    cdef iftSVM *iftCreatePreCompSVC(double C)
    cdef void iftDestroySVM(iftSVM *svm)
    
    cdef iftDataSet *iftKernelizeDataSet2(iftDataSet *Zref, iftDataSet *Zin, int kFunction, uchar traceNormalize, float *ktrace)
     
    cdef void iftSVMTrainOVA(iftSVM *svm, iftDataSet *Ztrain)
    cdef float** iftSVMLinearClassifyOVAPredMatrix(iftSVM* svm, iftDataSet *Ztest, iftDataSet* Ztrain, uchar sampleStatus)
    cdef float** iftSVMLinearClassifyOVA4(iftSVM* svm, iftDataSet *Z, iftSVMHyperplane **hyperplanes, uchar sampleStatus)

# constants extracted of enum in libsvm/svm.h
cdef enum:
    LINEAR = 0
    RBF = 2
    PRECOMPUTED = 4
