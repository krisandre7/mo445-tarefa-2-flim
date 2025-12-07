import pyift as ift
import sys
import os

if (len(sys.argv) != 5):
    print("python optimize-superpixel-param.py <superpixel path> <objmap path> <ground truth path> <results file>")
    exit(-1)

sup_path = sorted([os.path.join(sys.argv[1], x) for x in os.listdir(sys.argv[1])])
obj_path = sorted([os.path.join(sys.argv[2], x) for x in os.listdir(sys.argv[2])])
gt_path = sorted([os.path.join(sys.argv[3], x) for x in os.listdir(sys.argv[3])])

if len(sup_path) != len(obj_path) or len(sup_path) != len(gt_path):
    print("Given folders have different quantities of images")
    exit(-1)

t_list = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

file = open(sys.argv[4], "w")

file.write('threshold, assd, acc\n')

for threshold in t_list:
    acc = 0
    assd = 0
    for idx, img_path in enumerate(sup_path):
        
        print(img_path)

        sup = ift.ReadImageByExt(img_path)
        obj = ift.ReadImageByExt(obj_path[idx])
        gt  = ift.Normalize(ift.ReadImageByExt(gt_path[idx]), 0, 1)
        
        segm = ift.SuperPixelMajorityVote(sup, obj, threshold)
        
        assd += ift.ASSD(segm, gt)
        acc += ift.DiceSimilarity(segm, gt)
        
    acc /= len(sup_path)
    assd /= len(sup_path)
    print(threshold, acc, assd)
    file.write(str(threshold) + ',' + str(round(assd, 4)) + ',' + str(round(acc,4)) + '\n')

file.close()
