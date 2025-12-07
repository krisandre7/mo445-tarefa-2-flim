import pyift as ift
import sys
import os

if (len(sys.argv) != 5):
    print("python optimize-superpixel-param.py <original img path> <objmap path> <seeds path> <ground truth path> <results file>")
    exit(-1)

img_path = sorted([os.path.join(sys.argv[1], x) for x in os.listdir(sys.argv[1])])
obj_path = sorted([os.path.join(sys.argv[2], x) for x in os.listdir(sys.argv[2])])
seeds_path = sorted([os.path.join(sys.argv[3], x) for x in os.listdir(sys.argv[3])])
gt_path = sorted([os.path.join(sys.argv[4], x) for x in os.listdir(sys.argv[4])])

if len(img_path) != len(obj_path) or len(img_path) != len(gt_path) or len(seeds_path) != len(gt_path):
    print("Given folders have different quantities of images")
    exit(-1)

alphas = [x * 0.05 for x in range(20)]

file = open(sys.argv[4], "w")

file.write('alpha, assd, acc\n')

for alpha in alphas:
    acc = 0
    assd = 0
    for idx, img_path in enumerate(img_path):
        
        print(img_path)

        img = ift.ReadImageByExt(img_path)
        seeds = ift.ReadSeeds(seeds_path[idx], img)
        obj = ift.ReadImageByExt(obj_path[idx])
        gt  = ift.Normalize(ift.ReadImageByExt(gt_path[idx]), 0, 1)
        mimg = ift.ImageToMImage(img, ift.LABNorm_CSPACE)

        segm = ift.GraphCutWithObjmap(mimg, obj, seeds, alpha, 100)

        assd += ift.ASSD(segm, gt)
        acc += ift.DiceSimilarity(segm, gt)
        
    acc /= len(sup_path)
    assd /= len(sup_path)
    print(threshold, acc, assd)
    file.write(str(threshold) + ',' + str(round(assd, 4)) + ',' + str(round(acc,4)) + '\n')

file.close()
