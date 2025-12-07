#! /usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import sys
import random
import shutil


# script only valid to cropped mobio database
def split_cropped_database():
	# lê imagens da mobio com o seguinte formato: class_diretory_filename:
	# e.g.: 000093_02mobile_m416_02_r07_i0_0.png
	
	if len(sys.argv) != 4:
		print "Error. Please provide:"
		sys.exit("<Cropped_Mobio_Directory> <Output_Directory> <Number_images_per_video/subdirectory>")
		
	database_dir = sys.argv[1]
	output_path = sys.argv[2]
	nid = int(sys.argv[3])
	
	if not os.path.exists(database_dir):
		sys.exit("Error: Invalid Directory: {0}".format(database_dir))
	
	if os.path.exists(output_path):
		print "*** Cleaning the Output Directory: {0}***\n".format(output_path)
		[ os.remove(os.path.join(output_path, filename)) for filename in os.listdir(output_path) ]
	else:
		print "*** Creating the Output Directory: {0}***\n".format(output_path)
		os.makedirs(output_path)
		
	images = dict()
		
	nclasses = 0
	# for each file in directory
	for img_name in os.listdir(database_dir):
 		class_num = int(img_name.split("_")[0])
 		dir_key = img_name.split("_")[1] # get the video directory of the image in original mobio 
# 		print img_name
# 		print class_num
# 		print dir_key + "\n"
		
		if images.get(class_num) is None:
			images[class_num] = dict()
 			nclasses += 1
 		
		if images[class_num].get(dir_key) is None:
			images[class_num][dir_key] = []
 		
		images[class_num][dir_key].append(os.path.join(database_dir, img_name))
	
	count = 0
	# for each class
	for class_num in images:
		for dir_key in images[class_num]:
			random.shuffle(images[class_num][dir_key])
			
			# copy selected images
			for img in images[class_num][dir_key][:nid]:
				new_filename = img.split("/")[-1]
				new_filename = os.path.join(output_path, new_filename)
				print "class: {0}/{1} *** Copying: {2} -> {3}\n".format(str(int(class_num)), nclasses, img, new_filename)
				os.system("cp {0} {1};".format(img, new_filename))
				count += 1
		print ""
	print "*** Number of selected images: {0}\n".format(count)
		
	print "Done..."
		
split_cropped_database()
