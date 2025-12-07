'''
 Extracts Overall Performance Evaluation
'''

# Imports
import glob
import sys
import numpy as np
import argparse

# Create the arguments
parser = argparse.ArgumentParser(description='Extracts the overall results of an over-segmentation method');
parser.add_argument("-d", "--dir" , action = "store", dest = "dir_path", 
										required = True, help = "Dir containing the eval files");
args = parser.parse_args()

# Directory containing the evaluation files
files = glob.glob(args.dir_path + "/eval_*.txt");

if( len(files) > 0 ):
	# Order files in numerical order (0,1,...,9,10,11,...)
	ord_files = sorted(files, key = lambda l: int(l.split("_")[1].split(".")[0]))

	# Create overall evaluation file
	ovall_file = open(args.dir_path + "/eval.txt", 'w');

	# Header
	ovall_file.write("k,BR,UE,OVLAP,WBR,WUE\n");

	# For each evaluation file
	for f in ord_files :
		# Read data (skip the IDs and the header)
		data = np.loadtxt(f, usecols=range(1,4),skiprows=1, delimiter=",");
		
		k = int(f.split("_")[1].split(".")[0]);

		# Calculate the mean stats
		mean_br = np.mean(data[:,0]);
		mean_ue = np.mean(data[:,1]);
		mean_ovlap = np.mean(data[:,2]);
		mean_wbr = np.mean(data[:,0] * data[:,2])
		mean_wue = np.mean(data[:,1] / data[:,2])

		# Write the overall for that K
		ovall_file.write(str(k) + "," + str(mean_br) + "," + str(mean_ue) + "," + str(mean_ovlap) + "," + str(mean_wbr) + "," + str(mean_wue) + "\n");

	ovall_file.close();
else :
	print "ERROR: No eval files were found in <%s>!" % args.dir_path