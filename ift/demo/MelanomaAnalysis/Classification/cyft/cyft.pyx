cimport cyft
import numpy as np

def dataset_to_Xy(filename):
    cdef cyft.iftDataSet* dataset = cyft.iftReadOPFDataSet(filename)
    X = np.zeros((dataset.nsamples,dataset.nfeats))
    y = np.zeros(dataset.nsamples)
    
    cdef int s,f
    
    for s in range(dataset.nsamples):
        for f in range(dataset.nfeats):
            X[s,f] = dataset.sample[s].feat[f]
        y[s] = dataset.sample[s].truelabel
      
    cyft.iftDestroyDataSet(&dataset)      
    return X,y
