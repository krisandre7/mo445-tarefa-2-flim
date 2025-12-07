#!/usr/bin/python


import sys
from os.path import *
import os
import argparse as ap
import operator
import json

def main():
        parser = ap.ArgumentParser()
	parser.add_argument('--input_results_dir', required = True)
	parser.add_argument('--input_fold_dir', required = True)
	parser.add_argument('--label_name', required=  True)
        parser.add_argument('--accuracy_type', help = '0 - Dice, 1 - ASSD, 2 - IFT_BOTH', required = True, type=int)
        parser.add_argument('--marker_limit', required = False, default = -1, type=int, help = 'Computes accuracy considering only the results achieved with at most the given number of markers')

	args = vars(parser.parse_args())
	fold_dir = args['input_fold_dir']
	results_dir = args['input_results_dir']
	label_name = args['label_name']
        accuracy_type = args['accuracy_type']
	marker_limit = args['marker_limit']

	fold_dir_list = map(lambda y: abspath(join(fold_dir, y)), os.listdir(fold_dir))

	results_dir_list = map(lambda y: abspath(join(results_dir,y)), os.listdir(results_dir))
	mapped_labels_dirs = []

	for i in fold_dir_list:
		if isdir(i):
                        d = os.listdir(join(fold_dir, i))
			if 'mapped_labels' in d:
				mapped_labels_dirs.append(abspath(join(fold_dir, i, 'mapped_labels', label_name)))

        gt_labels = reduce(operator.add, [map(lambda y: join(x,y), os.listdir(x)) for x in mapped_labels_dirs])

        results_label = []
        
        for x in results_dir_list:
                label_files = sorted(filter(lambda x: x.find('label') >= 0, os.listdir(x)))
                marker_files = sorted(filter(lambda x: x.find('markers') >= 0, os.listdir(x)))

                idx = -1
                # If the marker limit is set, we search for the label result obtained by using at most the given amount of markers
                if marker_limit >= 1:
                        for i,y in enumerate(marker_files):
                                json_markers = json.load(open(join(x,y), 'r'))
                                marker_count = int(json_markers['marker_count'])
                                if marker_count <= marker_limit:
                                        idx = i
                                
                results_label.append(join(x, label_files[idx]))


        for gt in sorted(gt_labels, key=lambda x: basename(x)):
                idx = -1
                # Finding the segmentation result for the given ground truth
                for i,x in enumerate(results_label):
                        if x.find(basename(splitext(gt)[0])) >=0:
                                idx = i
                         
                if idx >= 0:
                        # If the segmentation for the current ground truth is available, we compute the segmentation accuracy
                        lb = results_label[idx]
                        print basename(lb), basename(gt)
                        os.system('iftSegmentationErrors %s %s %d' %(lb, gt, accuracy_type))

if __name__ == '__main__':
	main()	
