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


class CustomLinearSVM(object):
    def __init__(self, X_train, y_train, normalize_data=False, C=1e5, tol=1e-5):
        self.linear_svm     = svm.SVC(C=C, tol=tol, kernel="precomputed")
        if normalize_data:
            # normalize the dataset
            self.scaler  = StandardScaler()
            self.X_train = self.scaler.fit_transform(X_train)
        else:
            self.scaler  = None
            self.X_train = X_train
        self.y_train      = y_train
        self.kernel_train = None


    def fit(self):
        if not self.kernel_train:
            self.kernel_train = np.dot(self.X_train, self.X_train.T)
        self.linear_svm.fit(self.kernel_train, self.y_train)


    def decision_function(self, kernel):
        # it assumes that kernel is the kernel from the data already normalized by the scaler
        dists = self.linear_svm.decision_function(kernel)
        
        # gets the distances with respect to the first and only hyperplane 
        return dists




def random_NM(X, neg_train_set, pos_train_set, max_NM_time, normalize_data, log):
    print("*********** RANDOM MINING *************")
    log.write("*********** RANDOM MINING *************\n")

    log.write("- pos_train_set.size: {0}\n".format(pos_train_set.size))
    log.write("- neg_train_set.size: {0}\n".format(neg_train_set.size))

    t          = time.time() # point the time
    X_train    = np.concatenate((X[pos_train_set], X[neg_train_set]))
    y_train    = np.concatenate((np.array([1] * pos_train_set.size), np.array([-1] * neg_train_set.size)))
    custom_svm = CustomLinearSVM(X_train, y_train, normalize_data)

    print("- Training Linear SVM")
    log.write("- Training Linear SVM\n")
    custom_svm.fit()

    spent_time = time.time()-t

    if spent_time > max_NM_time:
        Error.print_error("Max NM time was exceeded: %f/%f" % (spent_time, max_NM_time), "NM.random_NM")

    log.write("--> Spent Time: %.2f secs\n" % spent_time)
    log.write("***************************************\n\n")
    log.flush()
    print("***************************************\n")

    return custom_svm, spent_time



def our_NM_old(X, neg_train_set_i, neg_validation_set_i, pos_train_set_i,
    max_NM_time, log):
    print("*********** OUR NEGATIVE MINING *************")
    log.write("*********** OUR NEGATIVE MINING *************\n")

    neg_train_set      = np.array(neg_train_set_i)
    neg_validation_set = np.array(neg_validation_set_i)
    pos_train_set      = np.array(pos_train_set_i)
    custom_svm_out     = None
    proc_time          = 0.0

    #****************************************************************************#
    it = 0
    while (proc_time < max_NM_time):
        log.write("----------------------------------\n")
        print("\n---> It: {2} - Current Time: {0}/{1} secs".format(proc_time, max_NM_time, it))
        log.write("---> It: {2} - Current Time: {0}/{1} secs\n".format(proc_time, max_NM_time, it))

        log.write("- pos_train_set.size: {0}\n".format(pos_train_set.size))
        log.write("- neg_train_set.size: {0}\n".format(neg_train_set.size))
        log.write("- neg_validation_set.size: {0}\n\n".format(neg_validation_set.size))
        log.flush()

        t = time.time() # points the time

        orig_train_ids = np.concatenate((pos_train_set, neg_train_set)) # idxs from X
        X_train        = np.concatenate((X[pos_train_set], X[neg_train_set]))
        y_train        = np.concatenate(([1]*pos_train_set.size, [-1]*neg_train_set.size))
        custom_svm     = CustomLinearSVM(X_train, y_train)

        X_test = X[neg_validation_set]
        y_test = np.array([-1]*neg_validation_set.size)

        # log.write("- neg_train_set.size: {0}\n".format(neg_train_set.size))
        # log.write("- pos_train_set.size: {0}\n".format(pos_train_set.size))
        # log.write("- neg_validation_set.size: {0}\n\n".format(neg_validation_set.size))
        # log.write("- X_train.shape: {0}\n".format(custom_svm.X_train.shape))
        # log.write("- y_train.shape: {0}\n".format(custom_svm.y_train.shape))
        # log.write("- X_test.shape: {0}\n".format(X_test.shape))
        # log.write("- y_test.shape: {0}\n\n".format(y_test.shape))

        print("- Training Linear SVM")
        log.write("- Training Linear SVM\n")
        custom_svm.fit()

        proc_time += time.time()-t # points the time
        
        ################# STOP CONDITION CHECKING ##################
        if proc_time <= max_NM_time:
            it += 1 # one valid training
            custom_svm_out = custom_svm
            n_neg_SVs = custom_svm.linear_svm.n_support_[0]

            if (n_neg_SVs == neg_train_set.size):
                print("** Stop Condition: All Neg. Train. samples are SVs\n")
                log.write("** Stop Condition: All Neg. Train. samples are SVs\n")
                log.write("----------------------------------\n")
                log.flush()
                break
        else:
            if (it == 0):
                Error.print_error("--> No SVM was trained - the first training exceeded the Max Time: %f/%f secs\n" % (proc_time, max_NM_time), "NM.our_NM")
            else:
                print("--> Ignore this iteration - the training exceeded the Max NM Time: %f/%f secs\n" % (proc_time, max_NM_time))
                log.write("\n--> Ignore this iteration - the training exceeded the Max NM Time: %f/%f secs\n" % (proc_time, max_NM_time))
                log.write("----------------------------------\n")
                break
        ############################################################


        t = time.time() # points the time

        SVs        = custom_svm.linear_svm.support_ # Pos and Neg SV idxs from X_train
        # just the Neg SV idxs from X_train - eliminates the Pos SV idxs
        neg_SVs    = np.array(list(set(SVs) - set(range(pos_train_set.size))), dtype="int")
        neg_SV_ids = orig_train_ids[neg_SVs] # idxs from the Neg SV with respect to X

        # just the Neg non-SV idxs from X_train
        neg_non_SVs    = np.array(list(set(range(pos_train_set.size, pos_train_set.size+neg_train_set.size)) - set(neg_SVs)), dtype="int")
        # idxs from the Neg non-SV with respect to X
        neg_non_SV_ids = orig_train_ids[neg_non_SVs]
        max_swaps      = neg_non_SV_ids.size # max number of swaps (num of neg non-SV)

        log.write("- neg_SVs.size: {0}/{1}\n".format(neg_SVs.size, neg_train_set.size))
        log.write("- neg_non_SVs.size: {0}/{1}\n\n".format(neg_non_SVs.size, neg_train_set.size))
       
        print("- Computing Neg. Non-SVs distances")
        neg_train_dists = custom_svm.decision_function(custom_svm.kernel_train[neg_non_SVs])

        print("- Computing Neg. Test distances")
        X_test         = custom_svm.scaler.transform(X_test) # normalize dataset
        kernel_test    = np.dot(X_test, custom_svm.X_train.T)
        neg_test_dists = custom_svm.decision_function(kernel_test)

        # list of tuples: (hyperplane_dist, id), where id is the idx with respect to X
        neg_dists = zip(neg_train_dists, neg_non_SV_ids) + zip(neg_test_dists, neg_validation_set)
        neg_dists = sorted(neg_dists, reverse=True) # DESCENDING order (the most positive to the less positive samples)
        neg_dists = np.array(neg_dists, dtype=np.object)
        
        new_neg_train_set     = np.concatenate((neg_SV_ids, neg_dists[:max_swaps,1]))
        n_swapped_neg_samples = len(set(new_neg_train_set) - set(neg_train_set))
        proc_time += time.time()-t # points the time

        ################# STOP CONDITION CHECKING ##################
        if proc_time <= max_NM_time:
            neg_train_set      = np.array(new_neg_train_set, dtype="int")
            neg_validation_set = np.array(neg_dists[max_swaps:,1], dtype="int")

            # stop condition - if no neg sample was swapped
            if (n_swapped_neg_samples == 0):
                print("** Stop Condition: No swaps\n")
                log.write("\n** Stop Condition: No swaps\n")
                log.write("----------------------------------\n")
                log.flush()
                break
        else:
            print("--> Stop: The swapping process exceeded the Max NM Time: %f/%f\n" \
                  "Return the Current Linear SVM" % (proc_time, max_NM_time))
            log.write("--> Stop: The swapping process exceeded the Max NM Time: %f/%f\n" \
                  "Return the Current Linear SVM\n" % (proc_time, max_NM_time))
            log.write("----------------------------------\n")            
            log.flush()
            break
        log.write("----------------------------------\n\n")
        log.flush()
        ############################################################
    #****************************************************************************#

    # Spent time by the method
    if proc_time <= max_NM_time:
        spent_time = proc_time
    else:
        spent_time = max_NM_time

    print("**************************************************\n")
    log.write("\n--> Total of valid NM iterations: %d\n" % it)
    log.write("--> Spent Time: %.2f\n" % spent_time)
    log.write("**************************************************\n\n")
    log.flush()

    return custom_svm_out, spent_time



def our_NM(X, neg_train_set_i, neg_validation_set_i, pos_train_set_i, max_NM_time, normalize_data, log):
    print("*********** OUR NEGATIVE MINING *************")
    log.write("*********** OUR NEGATIVE MINING *************\n")

    neg_train_set      = np.array(neg_train_set_i)
    neg_validation_set = np.array(neg_validation_set_i)
    pos_train_set      = np.array(pos_train_set_i)
    custom_svm_out     = None
    proc_time          = 0.0

    #****************************************************************************#
    it = 0
    while (proc_time < max_NM_time):
        log.write("----------------------------------\n")
        print("\n---> It: {2} - Current Time: {0}/{1} secs".format(proc_time, max_NM_time, it))
        log.write("---> It: {2} - Current Time: {0}/{1} secs\n".format(proc_time, max_NM_time, it))

        log.write("- pos_train_set.size: {0}\n".format(pos_train_set.size))
        log.write("- neg_train_set.size: {0}\n".format(neg_train_set.size))
        log.write("- neg_validation_set.size: {0}\n\n".format(neg_validation_set.size))
        log.flush()

        t = time.time() # points the time

        orig_train_ids = np.concatenate((pos_train_set, neg_train_set)) # idxs from X
        X_train        = np.concatenate((X[pos_train_set], X[neg_train_set]))
        y_train        = np.concatenate(([1]*pos_train_set.size, [-1]*neg_train_set.size))
        custom_svm     = CustomLinearSVM(X_train, y_train, normalize_data)


        # log.write("- neg_train_set.size: {0}\n".format(neg_train_set.size))
        # log.write("- pos_train_set.size: {0}\n".format(pos_train_set.size))
        # log.write("- neg_validation_set.size: {0}\n\n".format(neg_validation_set.size))
        # log.write("- X_train.shape: {0}\n".format(custom_svm.X_train.shape))
        # log.write("- y_train.shape: {0}\n".format(custom_svm.y_train.shape))
        # log.write("- X_test.shape: {0}\n".format(X_test.shape))
        # log.write("- y_test.shape: {0}\n\n".format(y_test.shape))

        print("- Training Linear SVM")
        log.write("- Training Linear SVM\n")
        custom_svm.fit()

        proc_time += time.time()-t # points the time
        print("proc_time: %f\n" % proc_time)

        ################# STOP CONDITION CHECKING ##################
        if proc_time <= max_NM_time:
            it += 1 # one valid training
            custom_svm_out = custom_svm
            n_neg_SVs = custom_svm.linear_svm.n_support_[0]

            if (n_neg_SVs == neg_train_set.size):
                print("** Stop Condition: All Neg. Train. samples are SVs\n")
                log.write("** Stop Condition: All Neg. Train. samples are SVs\n")
                log.write("----------------------------------\n")
                log.flush()
                break
        else:
            if (it == 0):
                Error.print_error("--> No SVM was trained - the first training exceeded the Max Time: %f/%f secs\n" % (proc_time, max_NM_time), "NM.our_NM")
            else:
                print("--> Ignore this iteration - the training exceeded the Max NM Time: %f/%f secs\n" % (proc_time, max_NM_time))
                log.write("\n--> Ignore this iteration - the training exceeded the Max NM Time: %f/%f secs\n" % (proc_time, max_NM_time))
                log.write("----------------------------------\n")
                break
        ############################################################


        t = time.time() # points the time
        
        SVs        = custom_svm.linear_svm.support_ # Pos and Neg SV idxs from X_train
        # just the Neg SV idxs from X_train - eliminates the Pos SV idxs
        neg_SVs    = np.array(list(set(SVs) - set(range(pos_train_set.size))), dtype="int")
        neg_SV_ids = orig_train_ids[neg_SVs] # idxs from the Neg SV with respect to X

        # just the Neg non-SV idxs from X_train
        neg_non_SVs    = np.array(list(set(range(pos_train_set.size, pos_train_set.size+neg_train_set.size)) - set(neg_SVs)), dtype="int")
        # idxs from the Neg non-SV with respect to X
        neg_non_SV_ids = orig_train_ids[neg_non_SVs]
        max_swaps      = neg_non_SV_ids.size # max number of swaps (num of neg non-SV)

        log.write("- neg_SVs.size: {0}/{1}\n".format(neg_SVs.size, neg_train_set.size))
        log.write("- neg_non_SVs.size: {0}/{1}\n\n".format(neg_non_SVs.size, neg_train_set.size))
       
        print("- Computing Neg. Non-SVs distances")
        neg_train_dists = custom_svm.decision_function(custom_svm.kernel_train[neg_non_SVs])

        print("- Computing Neg. Test distances")
        X_test = X[neg_validation_set]
        if custom_svm.scaler:
            X_test = custom_svm.scaler.transform(X_test) # normalize dataset
        kernel_test    = np.dot(X_test, custom_svm.X_train.T)
        neg_test_dists = custom_svm.decision_function(kernel_test)

        # list of tuples: (hyperplane_dist, id), where id is the idx with respect to X
        neg_dists = zip(neg_train_dists, neg_non_SV_ids) + zip(neg_test_dists, neg_validation_set)
        neg_dists = sorted(neg_dists, reverse=True) # DESCENDING order (the most positive to the less positive samples)
        neg_dists = np.array(neg_dists, dtype=np.object)
        
        new_neg_train_set     = np.concatenate((neg_SV_ids, neg_dists[:max_swaps,1]))
        n_swapped_neg_samples = len(set(new_neg_train_set) - set(neg_train_set))
        proc_time += time.time()-t # points the time

        ################# STOP CONDITION CHECKING ##################
        if proc_time <= max_NM_time:
            neg_train_set      = np.array(new_neg_train_set, dtype="int")
            neg_validation_set = np.array(neg_dists[max_swaps:,1], dtype="int")

            # stop condition - if no neg sample was swapped
            if (n_swapped_neg_samples == 0):
                print("** Stop Condition: No swaps\n")
                log.write("\n** Stop Condition: No swaps\n")
                log.write("----------------------------------\n")
                log.flush()
                break
        else:
            print("--> Stop: The swapping process exceeded the Max NM Time: %f/%f\n" \
                  "Return the Current Linear SVM" % (proc_time, max_NM_time))
            log.write("--> Stop: The swapping process exceeded the Max NM Time: %f/%f\n" \
                  "Return the Current Linear SVM\n" % (proc_time, max_NM_time))
            log.write("----------------------------------\n")            
            log.flush()
            break
        log.write("----------------------------------\n\n")
        log.flush()
        ############################################################
    #****************************************************************************#

    # Spent time by the method
    if proc_time <= max_NM_time:
        spent_time = proc_time
    else:
        spent_time = max_NM_time

    print("**************************************************\n")
    log.write("\n--> Total of valid NM iterations: %d\n" % it)
    log.write("--> Spent Time: %.2f\n" % spent_time)
    log.write("**************************************************\n\n")
    log.flush()

    return custom_svm_out, spent_time




def Felzenszwalb_NM_old(X, neg_train_set_i, neg_validation_set_i, pos_train_set_i,
    max_NM_time, log):
    print("*********** FELZENSZWALB NEGATIVE MINING *************")
    log.write("*********** FELZENSZWALB NEGATIVE MINING *************\n")

    neg_train_set      = np.array(neg_train_set_i)
    neg_validation_set = np.array(neg_validation_set_i)
    pos_train_set      = np.array(pos_train_set_i)
    custom_svm_out     = None
    proc_time          = 0.0

    #****************************************************************************#
    it = 0
    while (proc_time < max_NM_time):
        log.write("----------------------------------\n")
        print("\n---> It: {2} - Current Time: {0}/{1} secs".format(proc_time, max_NM_time, it))
        log.write("---> It: {2} - Current Time: {0}/{1} secs\n".format(proc_time, max_NM_time, it))

        log.write("- pos_train_set.size: {0}\n".format(pos_train_set.size))
        log.write("- neg_train_set.size: {0}\n".format(neg_train_set.size))
        log.write("- neg_validation_set.size: {0}\n\n".format(neg_validation_set.size))
        log.flush()

        t = time.time() # points the time

        orig_train_ids = np.concatenate((pos_train_set, neg_train_set)) # idxs from X
        X_train        = np.concatenate((X[pos_train_set], X[neg_train_set]))
        y_train        = np.concatenate(([1]*pos_train_set.size, [-1]*neg_train_set.size))
        custom_svm     = CustomLinearSVM(X_train, y_train)

        X_test = X[neg_validation_set]
        y_test = np.array([-1]*neg_validation_set.size)

        # log.write("- neg_train_set.size: {0}\n".format(neg_train_set.size))
        # log.write("- pos_train_set.size: {0}\n".format(pos_train_set.size))
        # log.write("- neg_validation_set.size: {0}\n\n".format(neg_validation_set.size))
        # log.write("- X_train.shape: {0}\n".format(custom_svm.X_train.shape))
        # log.write("- y_train.shape: {0}\n".format(custom_svm.y_train.shape))
        # log.write("- X_test.shape: {0}\n".format(X_test.shape))
        # log.write("- y_test.shape: {0}\n\n".format(y_test.shape))

        print("- Training Linear SVM")
        log.write("- Training Linear SVM\n")
        custom_svm.fit()

        proc_time += time.time()-t # points the time
        
        ################# STOP CONDITION CHECKING ##################
        if proc_time <= max_NM_time:
            it += 1 # one valid training
            custom_svm_out = custom_svm
            n_neg_SVs = custom_svm.linear_svm.n_support_[0]
        else:
            if (it == 0):
                Error.print_error("--> No SVM was trained - the first training exceeded the Max Time: %f/%f secs\n" % (proc_time, max_NM_time), "NM.our_NM")
            else:
                print("--> Ignore this iteration - the training exceeded the Max NM Time: %f/%f secs\n" % (proc_time, max_NM_time))
                log.write("\n--> Ignore this iteration - the training exceeded the Max NM Time: %f/%f secs\n" % (proc_time, max_NM_time))
                log.write("----------------------------------\n")
                break
        ############################################################


        t = time.time() # points the time
       
        print("- Computing Neg. Train. distances")
        neg_train_dists = custom_svm.decision_function(custom_svm.kernel_train)

        print("- Computing Neg. Test distances")
        X_test         = custom_svm.scaler.transform(X_test) # normalize dataset
        kernel_test    = np.dot(X_test, custom_svm.X_train.T)
        neg_test_dists = custom_svm.decision_function(kernel_test)

        # list of tuples: (hyperplane_dist, id), where id is the idx with respect to X
        neg_dists = zip(neg_train_dists, neg_train_set) + zip(neg_test_dists, neg_validation_set)
        neg_dists = sorted(neg_dists, reverse=True) # DESCENDING order (the most positive to the less positive samples)
        neg_dists = np.array(neg_dists, dtype=np.object)
        swap_mask = neg_dists[:,0] >= -1.00001 # get the samples with distances >= -1.00001
        
        new_neg_train_set     = neg_dists[swap_mask,1] # get the new neg train. ids
        new_neg_validation_set = neg_dists[~swap_mask,1] # get the new validation train. ids
        proc_time += time.time()-t # points the time

        ################# STOP CONDITION CHECKING ##################
        if proc_time <= max_NM_time:
            if (set(new_neg_train_set).issubset(set(neg_train_set))):
                print("** Stop Condition: No New Hard Negative Sample.\n")
                log.write("\n** Stop Condition: No New Hard Negative Sample.\n")
                log.write("----------------------------------\n")
                log.flush()
                break
            else:
                neg_train_set      = np.array(new_neg_train_set, dtype="int")
                neg_validation_set = np.array(new_neg_validation_set, dtype="int")
        else:
            print("--> Stop: The swapping process exceeded the Max NM Time: %f/%f\n" \
                  "Return the Current Linear SVM" % (proc_time, max_NM_time))
            log.write("--> Stop: The swapping process exceeded the Max NM Time: %f/%f\n" \
                  "Return the Current Linear SVM\n" % (proc_time, max_NM_time))
            break
        log.write("----------------------------------\n\n")
        log.flush()
        ############################################################
    #****************************************************************************#

    # Spent time by the method
    if proc_time <= max_NM_time:
        spent_time = proc_time
    else:
        spent_time = max_NM_time

    print("**************************************************\n")
    log.write("\n--> Total of valid NM iterations: %d\n" % it)
    log.write("--> Spent Time: %.2f\n" % spent_time)
    final_train_perc = 100.0*(custom_svm_out.X_train.shape[0]-pos_train_set.size)/(neg_train_set_i.size+neg_validation_set_i.size) 
    log.write("--> Final Neg. Train. set = %d samples\n" % (custom_svm_out.X_train.shape[0]-pos_train_set.size))
    log.write("--> Final Neg. Train. Set: %.2f%% from the Large Neg. Set.\n" 
              % final_train_perc)
    log.write("**************************************************\n\n")
    log.flush()

    return custom_svm_out, spent_time


def Felzenszwalb_NM(X, neg_train_set_i, neg_validation_set_i, pos_train_set_i, max_NM_time, normalize_data, log):
    print("*********** FELZENSZWALB NEGATIVE MINING *************")
    log.write("*********** FELZENSZWALB NEGATIVE MINING *************\n")

    neg_train_set      = np.array(neg_train_set_i)
    neg_validation_set = np.array(neg_validation_set_i)
    pos_train_set      = np.array(pos_train_set_i)
    custom_svm_out     = None
    proc_time          = 0.0

    #****************************************************************************#
    it = 0
    while (proc_time < max_NM_time):
        log.write("----------------------------------\n")
        print("\n---> It: {2} - Current Time: {0}/{1} secs".format(proc_time, max_NM_time, it))
        log.write("---> It: {2} - Current Time: {0}/{1} secs\n".format(proc_time, max_NM_time, it))

        log.write("- pos_train_set.size: {0}\n".format(pos_train_set.size))
        log.write("- neg_train_set.size: {0}\n".format(neg_train_set.size))
        log.write("- neg_validation_set.size: {0}\n\n".format(neg_validation_set.size))
        log.flush()

        t = time.time() # points the time

        orig_train_ids = np.concatenate((pos_train_set, neg_train_set)) # idxs from X
        X_train        = np.concatenate((X[pos_train_set], X[neg_train_set]))
        y_train        = np.concatenate(([1]*pos_train_set.size, [-1]*neg_train_set.size))
        custom_svm     = CustomLinearSVM(X_train, y_train, normalize_data)


        # log.write("- neg_train_set.size: {0}\n".format(neg_train_set.size))
        # log.write("- pos_train_set.size: {0}\n".format(pos_train_set.size))
        # log.write("- neg_validation_set.size: {0}\n\n".format(neg_validation_set.size))
        # log.write("- X_train.shape: {0}\n".format(custom_svm.X_train.shape))
        # log.write("- y_train.shape: {0}\n".format(custom_svm.y_train.shape))
        # log.write("- X_test.shape: {0}\n".format(X_test.shape))
        # log.write("- y_test.shape: {0}\n\n".format(y_test.shape))

        print("- Training Linear SVM")
        log.write("- Training Linear SVM\n")
        custom_svm.fit()

        proc_time += time.time()-t # points the time
        print("proc_time: %f\n" % proc_time)
        
        ################# STOP CONDITION CHECKING ##################
        if proc_time <= max_NM_time:
            it += 1 # one valid training
            custom_svm_out = custom_svm
            n_neg_SVs = custom_svm.linear_svm.n_support_[0]
        else:
            if (it == 0):
                Error.print_error("--> No SVM was trained - the first training exceeded the Max Time: %f/%f secs\n" % (proc_time, max_NM_time), "NM.our_NM")
            else:
                print("--> Ignore this iteration - the training exceeded the Max NM Time: %f/%f secs\n" % (proc_time, max_NM_time))
                log.write("\n--> Ignore this iteration - the training exceeded the Max NM Time: %f/%f secs\n" % (proc_time, max_NM_time))
                log.write("----------------------------------\n")
                break
        ############################################################


        t = time.time() # points the time
       
        print("- Computing Neg. Train. distances")
        neg_train_dists = custom_svm.decision_function(custom_svm.kernel_train)

        print("- Computing Neg. Test distances")
        X_test = X[neg_validation_set]
        if custom_svm.scaler:
            X_test = custom_svm.scaler.transform(X_test) # normalize dataset
        kernel_test    = np.dot(X_test, custom_svm.X_train.T)
        neg_test_dists = custom_svm.decision_function(kernel_test)

        # list of tuples: (hyperplane_dist, id), where id is the idx with respect to X
        neg_dists = zip(neg_train_dists, neg_train_set) + zip(neg_test_dists, neg_validation_set)
        neg_dists = sorted(neg_dists, reverse=True) # DESCENDING order (the most positive to the less positive samples)
        neg_dists = np.array(neg_dists, dtype=np.object)
        swap_mask = neg_dists[:,0] >= -1.00001 # get the samples with distances >= -1.00001
        
        new_neg_train_set     = neg_dists[swap_mask,1] # get the new neg train. ids
        new_neg_validation_set = neg_dists[~swap_mask,1] # get the new validation train. ids
        proc_time += time.time()-t # points the time

        ################# STOP CONDITION CHECKING ##################
        if proc_time <= max_NM_time:
            if (set(new_neg_train_set).issubset(set(neg_train_set))):
                print("** Stop Condition: No New Hard Negative Sample.\n")
                log.write("\n** Stop Condition: No New Hard Negative Sample.\n")
                log.write("----------------------------------\n")
                log.flush()
                break
            else:
                neg_train_set      = np.array(new_neg_train_set, dtype="int")
                neg_validation_set = np.array(new_neg_validation_set, dtype="int")
        else:
            print("--> Stop: The swapping process exceeded the Max NM Time: %f/%f\n" \
                  "Return the Current Linear SVM" % (proc_time, max_NM_time))
            log.write("--> Stop: The swapping process exceeded the Max NM Time: %f/%f\n" \
                  "Return the Current Linear SVM\n" % (proc_time, max_NM_time))
            break
        log.write("----------------------------------\n\n")
        log.flush()
        ############################################################
    #****************************************************************************#

    # Spent time by the method
    if proc_time <= max_NM_time:
        spent_time = proc_time
    else:
        spent_time = max_NM_time

    print("**************************************************\n")
    log.write("\n--> Total of valid NM iterations: %d\n" % it)
    log.write("--> Spent Time: %.2f\n" % spent_time)
    final_train_perc = 100.0*(custom_svm_out.X_train.shape[0]-pos_train_set.size)/(neg_train_set_i.size+neg_validation_set_i.size) 
    log.write("--> Final Neg. Train. set = %d samples\n" % (custom_svm_out.X_train.shape[0]-pos_train_set.size))
    log.write("--> Final Neg. Train. Set: %.2f%% from the Large Neg. Set.\n" 
              % final_train_perc)
    log.write("**************************************************\n\n")
    log.flush()

    return custom_svm_out, spent_time