cdef extern from "../libift/include/iftAdjSet.h":
    ctypedef struct iftAdjSet:
        int node
        float arcw
        iftAdjSet *next
