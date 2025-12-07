#! /usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import sys
import random
import shutil

# script only valid to MOBIO database
def split_MOBIO_database():
	if len(sys.argv) != 3:
		print "Error. Please provide:"
		sys.exit("<IMAGES_PNG location> <output_location>")
		
	database_dir = sys.argv[1]
	output_path = os.path.abspath(sys.argv[2])
		
	if not os.path.exists(database_dir):
		sys.exit("*** Error ***: Invalid directory \"{0}\" -> Try >= 0!".format(database_dir))
		
	# Create or clean the directory where will be save the image symbolic links
	if not os.path.exists(output_path): # database_path/pre_base
		print "*** Creating the Output Directory: {0}***\n".format(output_path)
		os.makedirs(output_path)
	else:
		print "*** Cleaning the train directory! ***\n"
		[ os.remove(os.path.join(output_path, filename)) for filename in os.listdir(output_path) ]
		
	images = dict() # dictionary of images with all images, separating by class
	
	print "*** Getting the name of all classes in Database ***\n"
	# initialize an empty list for each class
	for class_dir in os.listdir(database_dir):
		for c in os.listdir(os.path.join(database_dir, class_dir)):
			images[c] = dict()
			
	n = 0
	print "*** Getting all images ***\n"
	# save image_directory and image_name  in their class dict
	for dirname, dirnames, filenames in os.walk(database_dir):
		for filename in filenames:
			dir_key = dirname.split("/")[-1]
			dir_key = dir_key.split("_")[0] + dir_key.split("_")[1]
			class_key = dirname.split("/")[-2]
 			
			if images[class_key].get(dir_key) is None:
				images[class_key][dir_key] = []
				
			images[class_key][dir_key].append(os.path.join(dirname, filename))
			n += 1

	class_num = 1
	m = 0
	# for each class
	for class_key in images:
		for dir_key in images[class_key]:
			random.shuffle(images[class_key][dir_key])
 			
			# copy selected images
			for img in images[class_key][dir_key]:		
				m += 1
				new_filename = img.split("/")[-1]
				new_filename = "{0:06d}_{1}_".format(class_num, dir_key) + new_filename
				new_filename = os.path.join(output_path, new_filename)
				print "{0}/{1} *** Copying: {2} --> {3}\n".format(m, n, img, new_filename)
				os.system("cp {0} {1};".format(img, new_filename))
		print ""
		class_num += 1	
		
	print "Done..."
	

split_MOBIO_database()
