#! /usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import sys

def main():
	if len(sys.argv) != 2:
		print "Error. Please provide:"
		print "<database_location>"
		return 1
	
	database = sys.argv[1]
	files = [ f for f in os.listdir(database) if f.endswith(".jpg") ]
	
	print "Converting..."
	for f in files:
		filename = f.split("_")
		class_name = "0" + filename[0]
		image_number = "00" + filename[1].split(".")[0]
		filename = class_name + "_" + image_number + ".pgm"
		
		print "{0} --> {1}".format(f, filename)
		
		f = os.path.join(database, f)
		filename = os.path.join(database, filename)
		
		os.system("convert {0} -resize 200x200 {1};".format(f, filename))
		os.system("rm {0}".format(f))
	
main()