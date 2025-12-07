#!/usr/bin/python2.6
import os
import sys
import re
import time
import shutil

def getLabelName(filename):
	pure_name = filename.split(".")[0]
	label = int(pure_name.split("_")[0])
	unique_number = pure_name.split("_")[1]
	return label, pure_name, unique_number

def main ():	
	if len(sys.argv) < 3:
		print "Error. Please provide:"
		print "<database_location> <classes separated by spaces>"
		return 1
	database = sys.argv[1]
	classes_number = []
	for num in sys.argv[2:]:
		classes_number.append(int(num))


	EXTENSION = ".scn"
	INGORED_EXTENSIONS = ".DS_STORE .txt".split(" ")
	label_counter = {}
	output = []

	files = [ f for f in os.listdir(database) if f.endswith(EXTENSION)]
	files.sort()

	for f in files:
		label, pure_name, unique_number = getLabelName(f)
		if label not in label_counter:
			label_counter[label] = 0
		label_counter[label] += 1
		if label in classes_number:
			output.append(f)

	output.sort()
	if not os.path.exists(database+"subset"):
		os.makedirs(database+"subset")
	new_label = 1
	for f in output:
		label, pure_name, unique_number = getLabelName(f)
		new_name = "00000" + str(new_label) + "_" + unique_number + EXTENSION
		label_counter[label] -= 1
		if label_counter[label] == 0:
			new_label += 1
		print database+f, database+"subset/"+new_name
		shutil.copy(database+f, database+"subset/"+new_name)


	'''print "Arquivo gerado:"
	for key in dict_counter:
		print key, ":", dict_counter[key]'''

main ()
