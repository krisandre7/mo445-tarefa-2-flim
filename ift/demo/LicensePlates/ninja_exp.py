#!/usr/bin/python2.7
import os
import sys
import argparse
import time
import numpy as np

from subprocess import call
from multiprocessing import Pool

def run_experiment(exp_params):
	str_exp_params = [str(e) for e in exp_params]
	args = ["iftSelectCandidatesNinja"]
	args.extend(str_exp_params)
	print args
	#str_exp_params = " ".join([str(e) for e in exp_params])
	#print str_exp_params
	call(args)
	output_csv = str_exp_params[-1]
	metrics = np.genfromtxt(output_csv, delimiter=',').astype(np.float)
	print output_csv
	print metrics.shape
	return metrics

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("folds_path", type=str, help="Folds path")
	parser.add_argument("output_path", type=str, help="Output path")
	parser.add_argument("nproc", type=int, help="Number of processes")
	parser.add_argument("output_resume", type=str, help="Output resume")
	args = parser.parse_args()
   
	
	folds_path = args.folds_path
	output_path = args.output_path
	nproc = args.nproc
	output_resume = args.output_resume
	ncols = 7

	folds_path = os.path.abspath(folds_path)
	output_path = os.path.abspath(output_path)
	folds_dirs = [ d for d in os.listdir(folds_path) if not os.path.isfile(os.path.join(folds_path,d)) ]
	nfolds = len(folds_dirs)
	
	pool = Pool(processes=nproc)

	list_exp_params = []

	# create list of experiment configurations
	for k in range(nfolds):
		# create output folder
		outFoldDir = "{}/fold{}/".format(output_path, (k+1))
		if os.path.isdir(outFoldDir):
			print("Files exist in {} ".format(outFoldDir))
			return 0
		os.makedirs(outFoldDir)

		outTrainOrigDir = "{}/fold{}/{}/{}".format(folds_path, (k+1), "train", "orig")
		outTrainLabelDir = "{}/fold{}/{}/{}".format(folds_path, (k+1), "train", "label")
		outTestOrigDir = "{}/fold{}/{}/{}".format(folds_path, (k+1), "test", "orig")
		outTestLabelDir = "{}/fold{}/{}/{}".format(folds_path, (k+1), "test", "label")
		outputCSV = "{}/{}".format(outFoldDir,"resume.csv")

		exp_params = []
		exp_params.append(outTrainOrigDir)
		exp_params.append(outTrainLabelDir)
		exp_params.append(outTestOrigDir)
		exp_params.append(outTestLabelDir)		
		exp_params.append(outFoldDir)
		exp_params.append(outputCSV)	

		list_exp_params.append(exp_params)	

	# Run experiments	
	results = []
	if nproc == 1:
		for j in range(len(list_exp_params)):
			results.append(run_experiment(list_exp_params[j]) )
	else:
		results = pool.map(run_experiment, list_exp_params)
	
	print results

	results_arr = np.zeros((nfolds, ncols))
	for j in range(len(list_exp_params)):
		results_arr[j,:] = results[j]
	
	mean_results = np.mean(results_arr, axis=0)
	final_results = np.zeros((nfolds+1, ncols))
	final_results[:nfolds, :] = results_arr
	final_results[nfolds, :] = mean_results
	
	# write results
	if output_resume:
		np.savetxt(output_resume, final_results, fmt = '%.6f', delimiter = ',')
	
if __name__ == '__main__':
    main()
