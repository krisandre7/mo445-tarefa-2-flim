from cyift.dataset cimport *

cdef extern from "../libift/include/iftMetrics.h":
    
    # True positives, false positives, false negatives, true negatives.
    ctypedef struct iftErrorClassification:
        int tp
        int fp
        int fn
        int tn

    cdef float iftTruePositives(iftDataSet *Z)
    cdef float iftMeasureMCC(iftErrorClassification cm)
    cdef float iftCosineDistance(float *f1, float *f2, int n)
    cdef float iftCosineDistance2(float *f1, float *f2, int n)