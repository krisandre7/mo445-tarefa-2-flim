from cyift.dataset cimport *

cdef extern from "../libift/include/iftKmeans.h":
    cdef void iftKmeansRun(int bKmedoids, iftDataSet* Z, iftDataSet** pZk, int maxIterations, float minImprovement)
    cdef iftDataSet *iftKmeansInitCentroidsFromSamples(iftDataSet* Z, int k)
    cdef iftDataSet *iftKmeansInitCentroidsFromSamples2(iftDataSet* Z, int k)