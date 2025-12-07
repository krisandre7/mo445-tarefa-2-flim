cdef extern from "../libift/include/iftSet.h":
    ctypedef struct iftSet:
        int elem
        iftSet *next
        
    cdef int iftRemoveSet(iftSet **S)
    cdef void iftDestroySet(iftSet **S)
