import os
import sys

if (len(sys.argv) != 4):
    print("python decode.py <P1> <P2>")
    print("P1: initial layer")
    print("P2: final layer")
    print("P3: decoder id (e.g., 1,2,3)")
    exit()

init_layer  = int(sys.argv[1])    
final_layer = int(sys.argv[2])    
dec_id      = int(sys.argv[3])

for i in range(init_layer,final_layer+1):
    line = "iftDecodeLayer {} arch2D.json flim salie {}".format(i,dec_id)
    os.system(line)
    line = "iftSMansoniDelineation . salie {} 2 objs".format(i)
    os.system(line)
