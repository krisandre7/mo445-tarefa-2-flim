import os
import sys

if (len(sys.argv) != 5):
    print("python superpixel_dataset.py <P1> <P2> <P3> <P4>")
    print("P1: split")
    print("P2: number of superpixels")
    print("P3: superpixel folder")
    print("P4: 0 (unsupervised) 1 (supervised, image class) 2 (supervised, object label")
    exit()

split    = int(sys.argv[1])
nsuper   = int(sys.argv[2])
superdir = sys.argv[3]
mode     = int(sys.argv[4])

line = "rm -rf {} layer0".format(superdir)
os.system(line)

line = "iftConvertImagesToMImages train{}.csv layer0".format(split)
os.system(line)

f = open("train{}.csv".format(split),"r")
for line in f:
    filein    = line.strip()
    basename  = line.split("/")[1].split(".")[0]
    ext       = line.split("/")[1].split(".")[1].strip()
    print("Processing {}".format(basename))
    fileout   = "{}/{}.{}".format(superdir,basename,ext)    
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

#    if (os.path.exists("./masks")):
#        line = "RunSICLE --img {} --out {} --mask {} --n0 {} --nf {}".format(filein,fileout,filemask,5*nsuper,nsuper)
#    else:
#        line = "RunSICLE --img {} --out {} --n0 {} --nf {}".format(filein,fileout,5*nsuper,nsuper)
#    os.system(line)
    
f.close()

line = "iftSeedsFromSuperpixels {} seeds {}".format(superdir,mode)
os.system(line)

