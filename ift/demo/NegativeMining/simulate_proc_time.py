import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pdb
import sys
import time

if len(sys.argv) != 5:
	sys.exit("simulate_proc_time <n_train_pos> <n_train_neg> <n_validation_neg> <n_feats>")

n_train_pos = int(sys.argv[1])
n_train_neg = int(sys.argv[2])
n_val_neg   = int(sys.argv[3])
n_nfeats    = int(sys.argv[4])

print("- Generating the sets")
X_train = np.random.randn(n_train_pos+n_train_neg, n_nfeats)
y_train = np.concatenate(([1]*n_train_pos, [-1]*n_train_neg))
X_test  = np.random.randn(n_val_neg, n_nfeats)


proc_time = 0.0

t = time.time()
# scaler       = StandardScaler()
# print("- Normalizing X_train")
# X_train      = scaler.fit_transform(X_train)
print("- Dot")
kernel_train = np.dot(X_train, X_train.T)
print("- Training")
linear_svm = svm.SVC(C=1e6, kernel="precomputed")
linear_svm.fit(kernel_train, y_train)
proc_time = time.time()-t
print("--> Time: %f sec\n" % (proc_time))

# pdb.set_trace()

if (n_val_neg > 0):
	t = time.time()
	# print("- Normalizing X_test")
	# X_test = scaler.transform(X_test)
	print("- Computing Train Neg Distances")
	linear_svm.decision_function(kernel_train[:150])
	print("--> Time: %f sec\n" % (time.time()-t))

	t = time.time()
	print("- Computing Train Neg Distances")
	kernel_test = np.dot(X_test, X_train.T)
	linear_svm.decision_function(kernel_test)
	t2 = time.time()-t
	proc_time += t2
	print("--> Time: %f sec\n" % (t2))

print("--> Final Proc Time: %f sec\n" % (proc_time))

