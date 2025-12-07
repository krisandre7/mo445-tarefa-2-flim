import os
import sys
import pdb

if len(sys.argv) != 5:
    sys.exit("segmentation_errors <resul_labels_dir or result_labels.csv> <ground_truth_dir or ground_truth> <accuracy type: DICE - 0, ASSD - 1, IFT_BOTH - 2> <out_score_file.txt>")

# VALIDATIONS
if not os.path.exists(sys.argv[1]):
    sys.exit("File or Directory \"%s\" does not exist" % sys.argv[1])
if not os.path.exists(sys.argv[2]):
    sys.exit("File or Directory \"%s\" does not exist" % sys.argv[2])

# reading result label images
if (os.path.isdir(sys.argv[1])):
    result_labels_dir = sys.argv[1]
    result_labels = [os.path.join(result_labels_dir, img) for img in os.listdir(result_labels_dir)]
else:
    result_labels = open(sys.argv[1]).read().splitlines()
result_labels.sort()

# reading gt images
if (os.path.isdir(sys.argv[2])):
    gt_dir = sys.argv[2]
    gt_list = [os.path.join(gt_dir, img) for img in os.listdir(gt_dir)]
else:
    gt_list = open(sys.argv[2]).read().splitlines()
gt_list.sort()

accuracy_type = int(sys.argv[3])
out_score_filename = sys.argv[4]
if os.path.exists(out_score_filename):
    print 'Removing old scores file', out_score_filename
    os.remove(out_score_filename) # cleans the old score file

for result in result_labels:
    # getting basename without extension
    result_basename = os.path.basename(result)

    # finds the corresponding gt
    gt = None
    for gt_path in gt_list:
        if result_basename == os.path.basename(gt_path):
            gt = gt_path

    if gt:
        print(result_basename)
        os.system("echo %s >> %s" % (result_basename, out_score_filename))
        os.system("iftSegmentationErrors %s %s %d >> %s" % (result, gt, accuracy_type, out_score_filename))
        os.system("echo >> %s" % (out_score_filename))

