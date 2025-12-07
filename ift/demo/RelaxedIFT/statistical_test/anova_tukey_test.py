#!/usr/bin/python2.7
import os
import sys
import argparse
import numpy as np
import math
import csv
from scipy import stats
#from statsmodels.stats.multicomp import pairwise_tukeyhsd
from qsturng import qsturng, psturng

def computeF_ANOVA(listArr):
	mg = float(np.sum([np.sum(e) for e in listArr]))/np.sum([len(e) for e in listArr])

	ssbetween = np.sum([len(e)*((np.mean(e)-mg)**2) for e in listArr])
	sswithin = np.sum( [ (np.std(e)**2)*len(e)  for e in listArr] )
	
	dfbetween=len(listArr)-1
	dfwithin = np.sum([len(e) for e in listArr]) - len(listArr)

	msbetween = ssbetween/dfbetween
	mswithin = sswithin/dfwithin

	F = msbetween/mswithin
	P = stats.f.sf(F, dfbetween, dfwithin)
	
	return (F, P)
	

def computeHSD(listArr):
	mg = float(np.sum([np.sum(e) for e in listArr]))/np.sum([len(e) for e in listArr])

	ssbetween = np.sum([len(e)*((np.mean(e)-mg)**2) for e in listArr])
	sswithin = np.sum( [ (np.std(e)**2)*len(e)  for e in listArr] )
	
	dfbetween=len(listArr)-1
	dfwithin = np.sum([len(e) for e in listArr]) - len(listArr)

	msbetween = ssbetween/dfbetween
	mswithin = sswithin/dfwithin

	F = msbetween/mswithin

	k = len(listArr)
	q = qsturng(0.95, k, dfwithin)

	hsd = q * math.sqrt(mswithin/len(listArr[0]))
	
	return hsd
	

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("dir", type=str, help="Directory with results")
	parser.add_argument("delimiter", type=str, help="Delimiter between results")
	parser.add_argument("order", type=str, help="Highest first (h)| Lowest first (l)")
	parser.add_argument("confidence", type=float, help="Complemente of the interval of confidence. 0.05 means a 95p interval")
	parser.add_argument("-o", "--output", type=str, help="Output txt filename")
	args = parser.parse_args()
	
	directory = args.dir
	delimiter = args.delimiter
	order     = args.order
	output    = args.output
	confidence  = args.confidence

	# Read files in directory
	currentDir = ""
	if directory[0] != '/':
		currentDir = os.path.dirname(os.path.abspath(__file__))+"/"
	directory = currentDir + directory
	
	resFiles = [ f for f in os.listdir(directory) if f[0] != "."]
	resFiles.sort()

	# Read files
	numFiles = len(resFiles)
	resListArr = []
	for j in range(numFiles):
		pathResFile = directory + resFiles[j]
		resArr = np.genfromtxt(pathResFile, delimiter=delimiter)
		resListArr.append(resArr)
	
	# Run ANOVA
	f_value, p_value = stats.f_oneway(*resListArr)
	print "F val: {}".format(f_value)
	print "P val: {}".format(p_value)

	(F, P) = computeF_ANOVA(resListArr)
	print "F comp_val: {}".format(F)
	print "P comp_val: {}".format(P)

	if P < confidence:

		# Run Tukey
		# HSD resListArr
		HSD = computeHSD(resListArr)
		print "HSD val: {}".format(HSD)
		
		# Sort values by the mean
		listMean = [ np.mean(e) for e in resListArr]
		indexF   = np.argsort(listMean).astype(np.int)
		if order == 'h':
			indexF = indexF[::-1] # reverse
		
		numGroups = len(indexF)
		matrixCmp = np.zeros((numGroups,numGroups))
		for i in range(numGroups):
			for j in range(numGroups):
				if (math.fabs( listMean[indexF[i]] - listMean[indexF[j]] ) >  HSD):
					matrixCmp[i,j] = 1
				else:
					matrixCmp[i,j] = 0
		
		# Get differences list
		diffList = []
		for i in range(numGroups):
			for j in range(numGroups):
				if i < j:
					if matrixCmp[i,j] == 1:
						if order == 'l':
							elem = "{} < {}".format(resFiles[indexF[i]], resFiles[indexF[j]])
						else:
							elem = "{} > {}".format(resFiles[indexF[i]], resFiles[indexF[j]])
						diffList.append(elem)
					else:
						elem = "{} = {}".format(resFiles[indexF[i]], resFiles[indexF[j]])
						diffList.append(elem)	
		# Save diffList
		if output is not None:
			with open(output, 'w') as fp:
				a = csv.writer(fp)
				for k in range(len(diffList)):
					a.writerow([diffList[k]])
	else:
		print "There is no statistical significance between these samples in \
			the specified confidence interval"
	

if __name__ == '__main__':
	main()
