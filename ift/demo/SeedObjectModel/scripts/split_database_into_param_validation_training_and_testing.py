import os
import pdb
import random
import sys
from os.path import *


if len(sys.argv) < 6:
    sys.exit("split_database <all_imgs.csv> <perc_param_validation> <perc_train> <num_folds> <out_dir> [optional: previously_computed_parameter_validation_file.csv]")


all_imgs = []
with open(sys.argv[1]) as f:
    all_imgs = f.read().splitlines()
    f.close()
    
n_samples = len(all_imgs)

train_params_file = None
if len(sys.argv) > 6:
    print 'Reading previously computed parameter validation file'
    train_params_file = open(sys.argv[6], 'r')
    train_params_set = map(lambda x: x.strip(), train_params_file.readlines())
    n_train_params = len(train_params_set)
else:
    n_train_params = int(float(sys.argv[2]) * n_samples)

n_train = int(float(sys.argv[3]) * n_samples)
n_test  = n_samples - n_train
nfolds = int(sys.argv[4])
out_dir = sys.argv[5]

random.shuffle(all_imgs)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if train_params_file is None:
    # Separating the images for training the parameters into a disjoint set from the test candidates
    train_params_set = all_imgs[:n_train_params]
    test_complete_set  = all_imgs[n_train_params:]
    train_params_file = open(os.path.join(out_dir, "train_params.csv"), "w")
    train_params_file.write("\n".join(train_params_set))
else:
    test_complete_set = filter(lambda x: basename(x) not in map(basename, train_params_set), all_imgs)
    train_params_file = open(os.path.join(out_dir, "train_params.csv"), "w")
    train_params_file.write("\n".join(train_params_set))

for i in xrange(1,nfolds+1):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    random.shuffle(test_complete_set)
    
    test_set = test_complete_set[:n_test]
    train_set = filter(lambda x: basename(x) not in map(basename, test_set), all_imgs)

    if (set(map(basename, train_params_set)).intersection(set(map(basename, test_set))) != set([])) or \
       (set(map(basename, train_set)).intersection(set(map(basename, test_set))) != set([])):
        sys.exit("ERROR: Overlapping between sets")


    fold_dir = os.path.join(out_dir, "%02d" % i)
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)
    train_file = open(os.path.join(fold_dir, "train.csv"), "w")
    test_file  = open(os.path.join(fold_dir, "test.csv"), "w")
    
    train_file.write("\n".join(train_set))
    test_file.write("\n".join(test_set))

