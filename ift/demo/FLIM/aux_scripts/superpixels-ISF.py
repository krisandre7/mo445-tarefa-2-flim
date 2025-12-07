import os
import sys

if ((len(sys.argv) != 5) and (len(sys.argv) != 4)):
    print("python superpixels <P1> <P2> <P3> <P4>")
    print("P1: csv file with training images")
    print("P2: output folder with superpixel images")
    print("P3: number of superpixels")
    print("P4: input folder with object masks (optional)")
    exit()
    
if not os.path.exists(sys.argv[2]):
    os.mkdir(sys.argv[2])

f = open(sys.argv[1],"r")
for line in f:
    filein    = line.strip()
    basename  = line.split("/")[1].split(".")[0]
    ext       = line.split("/")[1].split(".")[1].strip()
    print("Processing {}".format(basename))
    fileout   = "{}/{}.{}".format(sys.argv[2],basename,ext)    
    if (len(sys.argv) == 5):
        filemask = "{}/{}.{}".format(sys.argv[4],basename,ext)
        os.system("iftISF_MIX_MEAN_Simple {} {} {} 0.5 12 10 2 {}".format(filein,fileout,int(sys.argv[3]),filemask))
    else:
        os.system("iftISF_MIX_MEAN_Simple {} {} {} 0.5 12 10 2".format(filein,fileout,int(sys.argv[3])))
        
f.close()

    
