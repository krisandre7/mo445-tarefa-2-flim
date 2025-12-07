#!/usr/bin/python2.6
import os
import sys
import re
import time
import random


def main ():	

	output = []
	output.append("N_SCALES, 3")
	output.append("N_LAYERS, 1")
#	output.append("%NO_POOLING,")
#	output.append("%NO_NORMALIZATION,")
	output.append("N_BANDS_INPUT, 1")
	output.append("SCALE_RATIO, 0.5, 0.5, 0.5")
	output.append("INPUT_NORM_SIZE, 9, 7, 5")
	output.append("N_KERNELS, 64, 128, 256")
	output.append("SIZE_KERNELS, 7, 5, 3")
	output.append("SIZE_POOLING, 7, 5, 3")	
	output.append("STRIDE, 1, 1, 1")
	output.append("ALPHA, 2, 2, 2")
	output.append("SIZE_NORM, 7, 5, 3")
	output.append("OUTPUT_NORM_SIZE, 0")
	output_fp = open("msconf.txt", "w")
	output_fp.write("\n".join(output))
	output_fp.close()

	print "Files saved"

main ()
