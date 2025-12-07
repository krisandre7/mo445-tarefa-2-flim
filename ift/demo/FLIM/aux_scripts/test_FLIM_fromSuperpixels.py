import os
import sys

if (len(sys.argv) != 4):
    print("python test_FLIM_fromSuperpixels <P1> <P2> <P3>")
    print("P1: last encoder layer")
    print("P2: split")
    print("P3: model folder")
    exit()

layer    = int(sys.argv[1])    
split    = int(sys.argv[2])
model    = sys.argv[3]

line = "rm -rf layer[0-{}] layer{}_test{}".format(layer,split,layer)
os.system(line)

line = "iftConvertImagesToMImages test{}.csv layer0".format(split)
os.system(line)

for i in range(1,layer+1):
    line = "iftEncodeMergedLayer arch2D.json {} {}".format(i,model)
    os.system(line)
    
line = "mv layer{} layer{}_test{}".format(layer,layer,split)
os.system(line)
line = "iftActivDataSet layer{}_test{} images 2 1".format(layer,split)
os.system(line)
line = "iftClassifyBySVM layer{}_test{}.zip svm_{}_{}.zip {}/results_{}_{}".format(layer,split,layer,split,model,layer,split)
os.system(line)
