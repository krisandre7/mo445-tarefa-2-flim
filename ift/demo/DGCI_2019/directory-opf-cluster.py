import pyift.pyift as ift
import sys
import os

if (len(sys.argv) != 7):
    print("python directory-superpixel.py <orig image path> <superpixel path> <output path> <number of neighbors> <Use coord as feat {0, 1}> <Use MST (1) Or Standard KnnGraph (0)>")
    exit(-1)

orig_path = sorted([os.path.join(sys.argv[1], x) for x in os.listdir(sys.argv[1])])
sup_path  = sorted([os.path.join(sys.argv[2], x) for x in os.listdir(sys.argv[2])])
if len(orig_path) != len(sup_path):
    print("Number of superpixels images differ from number of original images")
    exit(-1)

n = int(sys.argv[4])

for idx, img_path in enumerate(orig_path):
    
    out_path = os.path.join(sys.argv[3], os.path.splitext(os.path.basename(img_path))[0] + ".pgm")
    print(out_path)

    img = ift.ReadImageByExt(img_path)
    sup = ift.ReadImageByExt(sup_path[idx])
    A   = ift.Circular(1.5)
   
    feat = ift.ExtractImageFeatures(img, sup, A, bool(int(sys.argv[5])))
    Z = ift.MImageToDataSet(feat, sup)
    ift.SetStatus(Z, ift.IFT_TRAIN)

    if int(sys.argv[6]) == 1:
        mst = ift.CreateMST(Z)
        graph = ift.MSTtoKnnGraph(mst, n)
    elif int(sys.argv[6]) == 0:
        graph = ift.CreateKnnGraph(Z, n)
    else:
        print("Sixth argument must be 0 or 1")
        exit(-1)
    
    ift.UnsupTrain(graph, ift.NormalizedCutPtr())

    cluster = ift.DataSetToLabelImage(Z, sup, False, ift.IFT_GROUP)
    ift.WriteImageByExt(cluster, out_path)
