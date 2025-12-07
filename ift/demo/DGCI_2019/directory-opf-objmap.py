import pyift as ift
import sys
import os

if (len(sys.argv) != 6):
    print("python directory-superpixel.py <orig image path> <superpixel path> <cluster path> <seeds path> <output path>")
    exit(-1)

orig_path = sorted([os.path.join(sys.argv[1], x) for x in os.listdir(sys.argv[1])])
sup_path = sorted([os.path.join(sys.argv[2], x) for x in os.listdir(sys.argv[2])])
cluster_path  = sorted([os.path.join(sys.argv[3], x) for x in os.listdir(sys.argv[3])])
seeds_path = sorted([os.path.join(sys.argv[4], x) for x in os.listdir(sys.argv[4])])

if len(orig_path) != len(sup_path) or len(orig_path) != len(seeds_path) or len(orig_path) != len(sup_path):
    print("Number of superpixels images differ from number of original images")
    exit(-1)

for idx, img_path in enumerate(orig_path):

    out_path = os.path.join(sys.argv[5], os.path.splitext(os.path.basename(img_path))[0] + ".pgm")
    seeds_out_path = os.path.join(sys.argv[5], os.path.splitext(os.path.basename(img_path))[0] + ".txt")
    print(out_path)

    img = ift.ReadImageByExt(img_path)
    seeds = ift.ReadSeeds(img, seeds_path[idx])
    sup = ift.ReadImageByExt(sup_path[idx])
    cluster = ift.ReadImageByExt(cluster_path[idx])
    A = ift.Circular(1.5)
    
    feat = ift.ExtractImageFeatures(img, None, A, False)
    Z = ift.MImageToDataSet(feat, None)
    
    seeds = ift.SelectSeedsForEnhancement(seeds, cluster, 1, 0.80)
    ift.WriteSeeds(seeds, img, seeds_out_path)

    ift.LabelDataSetFromSeeds(Z, seeds, None)
    Ztrain = ift.ExtractSamples(Z, ift.IFT_SUPERVISED)
    ift.SetStatus(Ztrain, ift.IFT_TRAIN)
    graph = ift.CreateCplGraph(Ztrain)
    ift.SupTrain(graph)
    
    ift.SetStatus(Z, ift.IFT_TEST)
    ift.ClassifyWithCertaintyValues(graph, Z)
    objmap = ift.DataSetObjectMap(Z, None, 255, 2)

    ift.WriteImageByExt(objmap, out_path)
