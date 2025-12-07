import os
import pdb
import random
import sys



if len(sys.argv) != 6:
    sys.exit("split_database <all_imgs.csv> <perc_train> <perc_dev> <num_folds> <out_dir>")

nfolds = int(sys.argv[4])
all_imgs = []
with open(sys.argv[1]) as f:
    all_imgs = f.read().splitlines()
    f.close()
    
n_samples = len(all_imgs)

n_train = int(float(sys.argv[2]) * n_samples)
n_dev   = int(float(sys.argv[3]) * n_samples)
n_test  = n_samples - n_train - n_dev
out_dir = sys.argv[5]

random.shuffle(all_imgs)

test_set  = all_imgs[(n_train+n_dev):]

train_dev_imgs = list(all_imgs[:(n_train+n_dev)])

for i in xrange(1,nfolds+1):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # if The development set is not selected then we randomly shuffle all images 
    # for each fold before splitting into testing and training
    if n_dev <= 0:
        random.shuffle(all_imgs)
        test_set  = all_imgs[(n_train+n_dev):]
        train_dev_imgs = list(all_imgs[:(n_train+n_dev)])
    else:
        random.shuffle(train_dev_imgs)

    train_set = train_dev_imgs[:n_train]
    dev_set   = train_dev_imgs[n_train:]

    if (set(train_set).intersection(set(dev_set)) != set([])) or \
       (set(train_set).intersection(set(test_set)) != set([])) or \
       (set(dev_set).intersection(set(test_set)) != set([])):
        sys.exit("ERROR: Overlapping between sets")


    fold_dir = os.path.join(out_dir, "%02d" % i)
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    train_file = open(os.path.join(fold_dir, "train.csv"), "w")
    test_file  = open(os.path.join(fold_dir, "test.csv"), "w")

    if len(dev_set) > 0:
        dev_file   = open(os.path.join(fold_dir, "dev.csv"), "w")    
        dev_file.write("\n".join(dev_set))

    train_file.write("\n".join(train_set))
    test_file.write("\n".join(test_set))

