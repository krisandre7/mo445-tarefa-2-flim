cdef extern from "../libift/include/iftDescriptors.h":
    ctypedef struct iftFeatures:
        float *val
        int n
    
    cdef iftFeatures *iftCreateFeatures(int n)
    cdef iftFeatures *iftReadFeatures2(char *filename)
    cdef void iftWriteFeatures(iftFeatures *feat, char *filename)
    cdef void iftWriteFeatures2(iftFeatures *feat, char *filename)
    cdef void iftDestroyFeatures(iftFeatures **feat)
    
