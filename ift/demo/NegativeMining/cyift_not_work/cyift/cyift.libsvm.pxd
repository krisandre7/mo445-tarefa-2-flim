cdef extern from "../libsvm/svm.h":
    cdef struct svm_node:
        int index
        double value

    cdef struct svm_problem:
        int l
        double *y
        svm_node **x
     
    cdef struct svm_parameter:
        int svm_type
        int kernel_type
        int degree
        double gamma
        double coef0
        double cache_size
        double eps
        double C
        int nr_weight
        int *weight_label
        double* weight
        double nu
        double p
        int shrinking
        int probability
      
    cdef struct svm_model:
        svm_parameter param
        int nr_class
        int l
        svm_node **SV
        double **sv_coef
        double *rho
        double *probA
        double *probB
        int *sv_indices
        int *label
        int *nSV
        int free_sv
        
        
        
        
        
        
        
        
        
        
        