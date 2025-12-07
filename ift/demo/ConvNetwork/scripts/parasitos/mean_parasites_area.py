import os
import sys
import re
import Image
import numpy
import matplotlib
from matplotlib import pylab

def main ():	
	if len(sys.argv) != 2:
		print "Error. Please provide: (only .pgm files will be processed)"
		print "<parasite_database>"
		return 1
	database = sys.argv[1]
	#database = "../../bin/baseparasitos/"
	REGEX = r'[0-9]{1,3}_'
	files = [f for f in os.listdir(database) if f.endswith(".pgm")]

	number_classes = 0
	for f in files:
		image_class = re.search(REGEX, f, re.I)
		if image_class:
			image_class = image_class.group()
			image_class = image_class[:-1]
			image_class = int(image_class)
			if image_class > number_classes:
				number_classes = image_class

	list_class_pixels = [0 for i in range(number_classes)]

	for f in files:
		image_class = re.search(REGEX, f, re.I)
		if image_class:
			image_class = image_class.group()
			image_class = image_class[:-1]
			image_class = int(image_class)

			img = Image.open(database + f)
			img_data = list(img.getdata())

			counter = 0
			for i in range (len(img_data)):
				if img_data[i] == 1:
					counter +=1

			list_class_pixels[image_class-1] = counter

	print "Normalized:"
	
	counts = list_class_pixels
	counts = numpy.array(counts)/(max(counts) *1.)
	
	for i, value in enumerate(counts):
		print i+1, " : ", value

	threshold_suggestion = min([value for value in counts if value > 0.1])
	print "Threshold Suggestion:"
	print "Class 0, values < ", threshold_suggestion
	print "Class 1, otherwise"
	#pylab.plot(counts)
	#pylab.show()



main ()