import os
import sys

if (len(sys.argv) != 4):
    print("python train_flim_svm.py <P1> <P2> <P3>")
    print("P1: last encoder layer")
    print("P2: split")
    print("P3: number of superpixels")
    exit()

layer    = int(sys.argv[1])    
split    = int(sys.argv[2])
nsuper   = int(sys.argv[3])

line = "rm -rf bag flim layer[0-{}] layer{}_train{} superpixels train.csv seeds_files.txt".format(layer,layer,split)
os.system(line)

line = "ls -v train{}/* >> train.csv".format(split)
os.system(line)

line = "iftConvertImagesToMImages train{}.csv layer0".format(split)
os.system(line)

# create BoFP  

f = open("train.csv","r")
for line in f:
    filein    = line.strip()
    basename  = line.split("/")[1].split(".")[0]
    ext       = line.split("/")[1].split(".")[1].strip()
    print("Processing {}".format(basename))
    fileout   = "superpixels/{}.{}".format(basename,ext)    
    filemask  = "masks/{}.{}".format(basename,ext)
    if (os.path.exists("./masks")):
        line      = "iftDISF layer0/{}.mimg {} {} {} 255".format(basename,nsuper,fileout,filemask)
    else:
        line      = "iftDISF layer0/{}.mimg {} {}".format(basename,nsuper,fileout)        
    os.system(line)
#    if (os.path.exists("./masks")):
#        line      = "iftISF_MIX_MEAN_Simple {} {} {} 0.5 12 10 2 {}".format(filein,fileout,nsuper,filemask)
#    else:
#        line      = "iftISF_MIX_MEAN_Simple {} {} {} 0.5 12 10 2".format(filein,fileout,nsuper)        
#    os.system(line)
    
f.close()

line = "iftSeedsFromSuperpixels superpixels bag 1"
os.system(line)

line      = "ls -v bag >> seeds_files.txt"
os.system(line)

f = open("seeds_files.txt","r")

for line in f:
    file1 = line.strip()
    basename = file1.split("-")[0]
    file2 = "{}-fpts.txt".format(basename)
    line = "mv -f bag/{} bag/{}".format(file1,file2)
    os.system(line)

f.close()

# create and execute encoder

for i in range(1,layer+1):
    line = "iftCreateLayerModel bag arch2D.json {} flim".format(i)
    os.system(line)
    line = "iftMergeLayerModels arch2D.json {} flim".format(i)
    os.system(line)
    line = "iftEncodeMergedLayer arch2D.json {} flim".format(i)
    os.system(line)

# create dataset and train svm

line = "mv layer{} layer{}_train{}".format(layer,layer,split)
os.system(line)

line = "iftActivDataSet layer{}_train{} images 1 1".format(layer,split)
os.system(line)
line = "iftSupTrainBySVM layer{}_train{}.zip 0 0 1e2 0 svm_{}_{}.zip".format(layer,split,layer,split)
os.system(line)
