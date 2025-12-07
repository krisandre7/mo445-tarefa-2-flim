'''
 Plot Performance Comparison
'''
import glob
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import spline

# Constants
HEADERS = ["Number of Superpixels", "Boundary Recall", "Under-Segmentation Error", 
							 "Object Overlap", "Weighted Boundary Recall", "Weighted Under-Segmentation Error"];
ABBRV = ["K", "BR", "UE", "OVLAP", "wBR", "wUE"];
PLOT_ONLY = ["LSC", "SLIC", "ISF-MIX-MEAN", "ISF-MIX-ROOT", "OISF-GRID", "ISF-GRID-MEAN", "ISF-GRID-ROOT"]

# Create the arguments
parser = argparse.ArgumentParser(description="Plots the charts of the overall evaluation files within the directory");
parser.add_argument("-d", "--dir" , action = "store", dest = "dir_path", 
										required = True, help = "Dir containing the methods dirs");
parser.add_argument("-t", "--type" , action = "store", dest = "type", 
										required = True, help = "Type of evaluation [train, test, val, all,...]");
parser.add_argument("-s", "--savefigs" , action = "store", dest = "fig_path", 
										required = False, help = "Directory path to save the charts");
parser.add_argument("-f", "--format" , action = "store", dest = "fig_fmt", default="png",
										required = False, help = "Format of the chart images [png, jpg, pdf,...]");
args = parser.parse_args()

# List the methods within the dir
methods = sorted(glob.glob(args.dir_path + "/*"));

# Exists any method which was evaluated?
if( len(methods) > 0 ):

	# For every method
	for i in range(0,len(methods)):
		type_eval_dir = methods[i] + "/" + args.type;
		type_eval_file = glob.glob( type_eval_dir + "/eval.txt");
		method_name = methods[i].split("/")[-1]
		db_name = methods[i].split("/")[-2]

		# Exists an overall evaluation file?
		if( method_name in PLOT_ONLY ) :
			if( len(type_eval_file) > 0 ) :
				data = np.loadtxt(type_eval_file[0],skiprows=1, delimiter=",");	
				num_stats = data[0,:].shape[0]

				# Smooth curves
				x_sm = np.array(data[:,0])
				xsmooth = np.linspace(x_sm.min(), x_sm.max(), 300 )

				# For every stat
				for j in range(1, len(data[0,:])) :
					plt.figure(j)

					# Smooth curves 2
					y_sm = np.array(data[:,j])
					ysmooth = spline(x_sm, y_sm, xsmooth);

					plt.plot(xsmooth, ysmooth, label=method_name, linewidth=3)			
			else:
				print "ERROR: Non-existent type <%s>, or no overall evaluation file!" % type_eval_dir			

	# Config each chart
	for j in range(1, num_stats) :
		plt.figure(j)
		plt.xlabel(HEADERS[0])
		plt.ylabel(HEADERS[j])
		plt.title(db_name )# + " (" + args.type + ")")
		plt.legend()
		plt.grid(which="both", color="gray", alpha = 0.2)

		# Save charts in dir
		if( args.fig_path ):
			plt.savefig(args.fig_path + "/" + db_name + "_" + ABBRV[j] + "." + args.fig_fmt)

	# Show charts to user
	if( not args.fig_path ):
		plt.show();

else :
	print "ERROR: No methods were evaluated in <%s>!" % args.dir_path