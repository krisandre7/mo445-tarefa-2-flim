import os
import sys

if (len(sys.argv) != 5):
    print("python train_FLIM_BoFPdynamic_fromSuperpixels <P1> .. <P4>")
    print("P1: last encoder layer")
    print("P2: split")
    print("P3: number of filters per layer")
    print("P4: model folder")
    exit()

layer    = int(sys.argv[1])    
split    = int(sys.argv[2])
nfilters = int(sys.argv[3])
model    = sys.argv[4]

line = "rm -rf layer* train{}_layer{} {} bag".format(split,layer,model)
os.system(line)

line = "iftConvertImagesToMImages train{}.csv layer0".format(split)
os.system(line)

for i in range(1,layer+1):
    line = "iftPatchDataSetFromSeeds layer{} seeds arch2D.json {} {}".format(i-1,i,model)
    os.system(line)
    line = "iftFiltersFromPatchDataset {} arch2D.json {} {} images".format(model,i,nfilters)
    os.system(line)
    line = "iftEncodeMergedLayer arch2D.json {} {}".format(i,model)
    os.system(line)
    
line = "mv layer{} layer{}_train{}".format(layer,layer,split)
os.system(line)
line = "iftActivDataSet layer{}_train{} images 1 1".format(layer,split)
os.system(line)
line = "iftSupTrainBySVM layer{}_train{} 0 0 1e2 0 svm_{}_{}.zip".format(layer,split,layer,split)
os.system(line)
