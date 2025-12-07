"""
Negative Mining methods.
"""

# Author: Samuel Martins <sbm.martins@gmail.com>
# License: Unicamp
# Date: April, 23, 2015

from error import Error
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler

import pdb
import random
import sys
import time


# def CustomLinearSVM(object):
#     def __init__(self, C=1e5, tol=1e-5, kernel="precomputed"):
#         self.linear_svm    = svm.SVC(C=C, tol=tol, kernel=kernel)
#         self.X_train = []
#         self.y_train = []

#     def fit(X, y):
#         self.linear_svm



def our_NM(X, neg_train_set_i, neg_validation_set_i, pos_train_set_i,
    max_NM_time, log):
    print("*********** OUR NEGATIVE MINING *************")
    log.write("*********** OUR NEGATIVE MINING *************\n")

    neg_train_set      = np.array(neg_train_set_i)
    neg_validation_set = np.array(neg_validation_set_i)
    pos_train_set      = np.array(pos_train_set_i)
    linear_svm_out     = None
    proc_time          = 0.0

    it = 0
    while (proc_time < max_NM_time):
        log.write("----------------------------------\n")
        print("\n---> It: {2} - Current Time: {0}/{1} secs".format(proc_time, max_NM_time, it))
        log.write("---> It: {2} - Current Time: {0}/{1} secs\n".format(proc_time, max_NM_time, it))

        log.write("- pos_train_set: {0}\n".format(pos_train_set.size))
        log.write("- neg_train_set: {0}\n".format(neg_train_set.size))
        log.write("- neg_validation_set: {0}\n\n".format(neg_validation_set.size))

        t = time.time()

        orig_train_ids = np.concatenate((pos_train_set, neg_train_set)) # idxs from X
        X_train        = np.concatenate((X[pos_train_set], X[neg_train_set]))
        y_train        = np.concatenate((np.array([1] * pos_train_set.size), np.array([-1] * neg_train_set.size)))

        X_test = X[neg_validation_set]
        y_test = np.array([-1] * neg_validation_set.size)

        log.write("- neg_train_set: {0}\n".format(neg_train_set.size))
        log.write("- pos_train_set: {0}\n".format(pos_train_set.size))
        log.write("- neg_validation_set: {0}\n\n".format(neg_validation_set.size))
        log.write("- X_train.shape: {0}\n".format(X_train.shape))
        log.write("- y_train.shape: {0}\n".format(y_train.shape))
        log.write("- X_test.shape: {0}\n".format(X_test.shape))
        log.write("- y_test.shape: {0}\n\n".format(y_test.shape))
        log.flush()

        print("- Training Linear SVM")
        log.write("- Training Linear SVM\n")
        std_scaler = StandardScaler()
        X_train    = std_scaler.fit_transform(X_train)

        linear_svm   = svm.SVC(C=1e5, tol=1e-5, kernel="precomputed")
        kernel_train = np.dot(X_train, X_train.T)
        linear_svm.fit(kernel_train, y_train)

        SVs        = linear_svm.support_ # Pos and Neg SV idxs from X_train
        # just the Neg SV idxs from X_train - eliminates the Pos SV idxs
        neg_SVs    = np.array(list(set(SVs) - set(range(pos_train_set.size))), dtype="int")
        neg_SV_ids = orig_train_ids[neg_SVs] # idxs from the Neg SV with respect to X

        # just the Neg non-SV idxs from X_train
        neg_non_SVs    = np.array(list(set(range(pos_train_set.size, pos_train_set.size+neg_train_set.size)) - set(neg_SVs)), dtype="int")
        # idxs from the Neg non-SV with respect to X
        neg_non_SV_ids = orig_train_ids[neg_non_SVs]
        max_swaps      = neg_non_SV_ids.size # max number of swaps (num of neg non-SV)

        ################# STOP CONDITION CHECKING ##################
        if max_swaps == 0:
            print("** Stop Condition: All Neg. Train. samples are SV\n")
            log.write("\n** Stop Condition: All Neg. Train. samples are SV\n")
            log.write("----------------------------------\n")
            log.flush()
            linear_svm_out = linear_svm
            it += 1 # one valid iteration
            break
        ############################################################
        

        
        print("- Computing Neg. Non-SVs distances")
        neg_train_dists = linear_svm.decision_function(np.dot(X_train[neg_non_SVs], X_train.T))
        neg_train_dists = neg_train_dists[:,0] # gets the distances with respect to the first and only hyperplane 

        print("- Computing Neg. Test distances")
        X_test      = std_scaler.transform(X_test)
        kernel_test = np.dot(X_test, X_train.T)
        
        neg_test_dists = linear_svm.decision_function(kernel_test)
        neg_test_dists = neg_test_dists[:,0] # gets the distances with respect to the first and only hyperplane

        # list of tuples: (hyperplane_dist, id), where id is the idx with respect to X
        neg_dists = zip(neg_train_dists, neg_non_SV_ids) + zip(neg_test_dists, neg_validation_set)
        neg_dists = sorted(neg_dists, reverse=True) # DESCENDING order (the most positive to the less positive samples)
        neg_dists = np.array(neg_dists, dtype=np.object)

        new_neg_train_set     = np.concatenate((neg_SV_ids, neg_dists[:max_swaps,1]))
        n_swapped_neg_samples = len(set(new_neg_train_set) - set(neg_train_set))
        proc_time += time.time()-t # points the time
        
        if proc_time <= max_NM_time:
            it += 1 # one valid iteration
            neg_train_set      = np.array(new_neg_train_set, dtype="int")
            neg_validation_set = np.array(neg_dists[max_swaps:,1], dtype="int")

            # stop condition - if no neg sample was swapped
            if (n_swapped_neg_samples == 0):
                print("** Stop Condition: No swaps\n")
                log.write("\n** Stop Condition: No swaps\n")
                log.write("----------------------------------\n")
                log.flush()

                linear_svm_out = linear_svm
                break
        else:
            print("--> Ignore this iteration - it exceeded the Max NM Time: %f/%f\n" % (proc_time, max_NM_time))
            log.write("\n--> Ignore this iteration - it exceeded the Max NM Time: %f/%f\n" % (proc_time, max_NM_time))
            log.write("----------------------------------\n")
            
            if (it == 0):
                print("###\nInsufficient Max NM Time: the first iteration exceeded the Max NM Time\n" \
                      "It will be considered a Random Negative Mining\n###\n")
                log.write("###\nInsufficient Max NM Time: the first iteration exceeded the Max NM Time\n" \
                      "It will be considered a Random Negative Mining\n###\n")

                # get the first and only trained linear_svm which was learned with an initial random neg. train. set
                linear_svm_out = linear_svm
            break
        log.write("----------------------------------\n\n")
        log.flush()

    
    print("**************************************************\n")
    log.write("\n--> Total of valid NM iterations: %d\n" % it)
    log.write("**************************************************\n\n")
    log.flush()

    return neg_train_set, linear_svm_out




# cpdef test(np.ndarray[float, ndim=2] array, neg_train_set):
#     cdef float **data
#     data = <float**> malloc(array.shape[0] * sizeof(float*))
#     cdef int j

#     cdef iftDataSet *Z = NULL
#     Z = cydata.create_dataset(neg_train_set.size, array.shape[1], n_classes=2, alloc_feats=False)

#     # for i in xrange(array.shape[0]):
#     #     data[i] = &array[i,0]
#     for i in xrange(neg_train_set.size):
#         j = neg_train_set[i]
#         Z.sample[i].feat = &array[j,0]

#     print(array.shape[0], array.shape[1])
    
#     print("oi")
#     # print(array[0,1])
#     # print(Z.sample[0].feat[1])
#     