import os
import sys
    
if (len(sys.argv) != 4):
    print("python exec-flim.py <P1> <P2> <P3>")
    print("P1: number of layers (if negative, do not encode layers again)")
    print("P2: layer for the results")
    print("P3: model_type (0, 1, 2)")
    exit()

nlayers      = int(sys.argv[1])
target_layer = int(sys.argv[2])
model_type   = int(sys.argv[3])

number_of_points_per_marker = 1

os.system("iftMedianFiltering images 1.5 filtered")
line = "iftBagOfFeatPoints filtered markers {} bag".format(number_of_points_per_marker)
os.system(line)

for layer in range(1,nlayers+1):
    line = "iftCreateLayerModel bag arch.json {} flim".format(layer)
    os.system(line)
    if (model_type == 0):
        line = "iftEncodeLayer arch.json {} flim".format(layer)
        os.system(line)
    else:
        line = "iftMergeLayerModels arch.json {} flim".format(layer)
        os.system(line)
        line = "iftEncodeMergedLayer arch.json {} flim".format(layer)
        os.system(line)

line = "iftDecodeLayer {} arch.json flim {} salie".format(target_layer, model_type)
os.system(line)
        
