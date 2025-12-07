cdef extern from "../libift/include/iftFHeap.h":
    ctypedef struct iftFHeap:
        float *value
        char *color
        int *node
        int *pos
        int last
        int n
        char removal_policy
