"""
Negative Mining methods.
"""

# Author: Samuel Martins <sbm.martins@gmail.com>
# License: Unicamp
# Date: April, 28, 2015

from error import Error
import numpy as np
from NM import CustomLinearSVM

import pdb
import random
import sys
import time


def classify_by_SVM(X, pos_test_set, impostor_test_set, custom_svm, log):
    log.write("............... Final Classification ...............\n")

    log.write("- pos_test_set.size = %d\n" % pos_test_set.size)
    log.write("- impostor_test_set.size = %d\n" % impostor_test_set.size)
    log.write("- X_train.shape = {0}\n".format(custom_svm.X_train.shape))

    if custom_svm.scaler:
        X_pos_test      = custom_svm.scaler.transform(X[pos_test_set]) # normalize the pos test set
        X_impostor_test = custom_svm.scaler.transform(X[impostor_test_set]) # normalize the impostor test set
    else:
        X_pos_test      = X[pos_test_set]
        X_impostor_test = X[impostor_test_set]

    log.write("- X_pos_test.shape = {0}\n".format(X_pos_test.shape))
    log.write("- X_impostor_test.shape = {0}\n".format(X_impostor_test.shape))
    
    kernel_pos_test      = np.dot(X_pos_test, custom_svm.X_train.T)
    kernel_impostor_test = np.dot(X_impostor_test, custom_svm.X_train.T)
    pos_scores           = custom_svm.decision_function(kernel_pos_test)
    neg_scores           = custom_svm.decision_function(kernel_impostor_test)

    log.write("...................................................\n")

    return pos_scores, neg_scores


