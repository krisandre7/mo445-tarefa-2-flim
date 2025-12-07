import pyift as ift
import sys
import os

if (len(sys.argv) != 4):
    print("python directory-superpixel.py <orig image path> <out superpixel path> <number of superpixels>")
    exit(-1)

orig_path = sorted([os.path.join(sys.argv[1], x) for x in os.listdir(sys.argv[1])])

for img_path in orig_path:

    n = sys.argv[3]
    out_path = os.path.join(sys.argv[2], os.path.splitext(os.path.basename(img_path))[0] + ".pgm")
    print(out_path)
    os.system("iftRISF_segmentation -i " + img_path + " -o temp.pgm -n " + str(3 * int(n)) + " -s 1")
    os.system("iftRISF_segmentation -i " + img_path + " -o " + out_path + " -n " + n + " -s 4"+" -l temp.pgm")

    A = ift.Circular(1.5)
    img = ift.RelabelRegions(ift.ReadImageByExt(out_path), A)
    ift.WriteImageByExt(img, out_path)
