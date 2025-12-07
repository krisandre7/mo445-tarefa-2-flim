#!/usr/bin/python2.6
import os
import sys
import re
import time

def main ():	
	if len(sys.argv) != 4:
		print "Error. Please provide:"
		print "<database_location> <number_classes> <number_samples_per_class>"
		return 1
	database = sys.argv[1]
	number_classes = int(sys.argv[2])
	number_samples_per_class = int(sys.argv[3])

	INGORED_EXTENSIONS = ".DS_STORE .txt".split(" ")
	
	REGEX = r'[0-9]{1,3}_'
	dict_class_counter = {}
	dict_counter = {}
	output = []

	files = [ f for f in os.listdir(database) if f.endswith(".pgm")]
	files.sort()

	for f in files:
		image_class = re.search(REGEX, f, re.I)
		if image_class:	#Get the class from the file name
			image_class = image_class.group()
			image_class = image_class[:-1]
			image_class = int(image_class)
			if image_class not in dict_counter:
				dict_counter[image_class] = 1
			else:
				dict_counter[image_class] +=1
			if len(dict_class_counter) < number_classes:
				if image_class not in dict_class_counter:
					dict_class_counter[image_class] = 1

			if image_class in dict_class_counter:
				if dict_class_counter[image_class] <= number_samples_per_class:
					dict_class_counter[image_class] +=1
					output.append(str(image_class) + "," + f)

	output_fp = open(database+"filenames.txt", "w")
	output_fp.write("\n".join(output))
	output_fp.close()

	'''print "Arquivo gerado:"
	for key in dict_counter:
		print key, ":", dict_counter[key]'''
	
main ()
