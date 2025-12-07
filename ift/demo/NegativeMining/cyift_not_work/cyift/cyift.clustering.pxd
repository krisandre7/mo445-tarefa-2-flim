from cyift.adjset cimport *
from cyift.dataset cimport *
from cyift.fheap cimport *
from cyift.set cimport *

cdef extern from "../libift/include/iftClustering.h":
    ctypedef struct iftKnnNode:
        float maxarcw
        int root
        int sample
        iftAdjSet *adj
        iftSet *adjplat
        
    ctypedef struct iftKnnGraph:
        iftKnnNode *node
        float *pathval
        int *ordered_nodes
        int nnodes
        float *maxarcw
        int kmax
        int k
        iftFHeap *Q
        iftDataSet *Z
        
    ctypedef float (*iftKnnGraphCutFun)(iftKnnGraph *graph)
        
    cdef iftKnnGraph *iftCreateKnnGraph(iftDataSet *Z, int kmax)
    cdef void iftDestroyKnnGraph(iftKnnGraph **graph)
    cdef int iftUnsupTrain(iftKnnGraph *graph, iftKnnGraphCutFun iftGraphCutFun)
    
    cdef float iftKnnGraphCutFun1(iftKnnGraph *graph)
    cdef float iftKnnGraphCutFun2(iftKnnGraph *graph)
    cdef float iftNormalizedCut(iftKnnGraph *graph)
    cdef float iftKnnGraphCutFun4(iftKnnGraph *graph)
    cdef float iftKnnGraphCutFun5(iftKnnGraph *graph)
    
    cdef iftSet *iftGetKnnRootSamples(iftKnnGraph *graph)
    cdef iftSet *iftGetKnnBoundarySamples(iftKnnGraph *graph)
    cdef iftDataSet  *iftGraphBoundaryReduction(iftKnnGraph *graph, iftDataSet *Z)