import os
import sys

if (len(sys.argv) != 3):
    print("python classify.py <P1> <P2>")
    print("P1: layer")
    print("P2: folder with the original images")
    exit()

layer    = int(sys.argv[1])    
orig_dir = sys.argv[2]    
    
line = "iftActivDataSet layer{} {} 1 1".format(layer,orig_dir)
os.system(line)
line = "iftSplitDataSet layer{}.zip 0.5 3 3 data".format(layer)
os.system(line)

for i in range(1,4):
    line = "iftSupTrainBySVM data_train_00{}.zip 0 0 1e2 0 svm.zip".format(i)
    os.system(line)
    line = "iftClassifyBySVM data_test_00{}.zip svm.zip results".format(i)
    os.system(line)
