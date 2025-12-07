import os
import sys

if (len(sys.argv) != 3):
    print("python encode.py <P1> <P2>")
    print("P1: initial layer")
    print("P2: final layer")
    exit()

init_layer  = int(sys.argv[1])    
final_layer = int(sys.argv[2])    
    
for i in range(init_layer,final_layer+1):
    line = "iftCreateLayerModel bag arch2D.json {} flim".format(i)
    os.system(line)
    line = "iftMergeLayerModels arch2D.json {} flim".format(i)
    os.system(line)
    line = "iftEncodeMergedLayer arch2D.json {} flim".format(i)
    os.system(line)
    
