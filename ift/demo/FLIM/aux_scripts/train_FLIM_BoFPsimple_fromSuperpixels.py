# Verify variant that creates a bag for each layer from the markers
# obtained by FLIM cluster: this requires to execute patches from
# seeds, filters from patches that create markers, and bag from
# markers for each loop iteration

import os
import sys

if (len(sys.argv) != 5):
    print("python train_FLIM_BoFPsimple_fromSuperpixels <P1> .. <P4>")
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

line = "iftPatchDataSetFromSeeds layer0 seeds arch2D.json 1 {}".format(model)
os.system(line)

line = "iftFiltersFromPatchDataset {} arch2D.json 1 {} images".format(model,nfilters)
os.system(line)

line = "iftBagOfFeatPoints train{}.csv {}/markers1 3.0 1 1 bag".format(split,model)
os.system(line)

for i in range(1,layer+1):
    line = "iftCreateLayerModel bag arch2D.json {} {}".format(i,model)
    os.system(line)
    line = "iftMergeLayerModels arch2D.json {} {}".format(i,model)
    os.system(line)
    line = "iftEncodeMergedLayer arch2D.json {} {}".format(i,model)
    os.system(line)    

line = "mv layer{} layer{}_train{}".format(layer,layer,split)
os.system(line)
line = "iftActivDataSet layer{}_train{} images 1 1".format(layer,split)
os.system(line)
line = "iftSupTrainBySVM layer{}_train{} 0 0 1e2 0 svm_{}_{}.zip".format(layer,split,layer,split)
os.system(line)
