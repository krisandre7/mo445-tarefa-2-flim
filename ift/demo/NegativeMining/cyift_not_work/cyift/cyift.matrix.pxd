cdef extern from "../libift/include/iftMatrix.h":
    ctypedef struct iftMatrix:
        double *val
        int ncols
        int nrows
        int *tbrow
        int n