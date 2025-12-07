#!/usr/bin/python2.7
import os
import Image
import math
import sys

if len(sys.argv) != 2:
	print "Usage: python preprocess-parasites.py <database_location>"
	return 1

#DATABASE = "../../bin/baseparasitosp/"
DATABASE = sys.argv[1]
PPM_FILES = [f for f in os.listdir(DATABASE) if f.endswith('ppm')]
PPM_FILES.sort()
PGM_FILES = [f for f in os.listdir(DATABASE) if f.endswith('pgm')]
PGM_FILES.sort()
SMALL_PARASITE_CLASS = [1, 2, 3, 4, 5, 6, 12, 14]
BIG_PARASITE_CLASS = [7, 8, 9, 10, 11, 13]


def convertLinear2Matrix (position, xsize, ysize):
	x = (position % (xsize * ysize)) % xsize
	y = (position % (ysize * xsize)) / xsize
	return (x, y)

def getBB (img_data, xsize, ysize):
	listPixelsOnX = []
	listPixelsOnY = []
	for i, val in enumerate(img_data):
		if val == 1:
			listPixelsOnX.append(convertLinear2Matrix(i, xsize, ysize)[0])
			listPixelsOnY.append(convertLinear2Matrix(i, xsize, ysize)[1])
	return [(min(listPixelsOnX),min (listPixelsOnY)), (max(listPixelsOnX), max(listPixelsOnY))]

class BBsize ():
	def __init__(self, size, rectangle):
		self.size = size
		self.rectangle = rectangle

	def __repr__(self):
		return str(self.rectangle[0]) + ", " + str(self.rectangle[1]) + ": " + str(self.size)

def main():
	#size = [float("-inf"),float("-inf")]
	BBList = []
	size = BBsize(0, [(),()])
	largestSize = [0, 0]
	for f in PGM_FILES:	#COMPUTES THE MAXIMUM SIZE
		img = Image.open(DATABASE + f)
		gt_data = list(img.getdata())
		
		#BB is a list with two tuples. The positions of the rectangle in matricial position
		BB = getBB(gt_data, img.size[0], img.size[1])
		
		#SizeBB stores the number of pixels inside the bounding box
		sizeBB = math.fabs((BB[0][0] - BB[1][0])) * math.fabs((BB[0][1]) - BB[1][1])

		#Append the BB of each image in a list
		BBList.append(BBsize(sizeBB, BB))
		
		#Save the largest BB
		if sizeBB > size.size: size = BBsize(sizeBB, BB)
		largestSize[0] = max(largestSize[0], img.size[0])
		largestSize[1] = max(largestSize[1], img.size[1])



	print "Biggest: ", size
	for f in range(len(PPM_FILES)):


		img = Image.open(DATABASE + PPM_FILES[f])
		#gt = Image.open(DATABASE + PGM_FILES[f])
		current_BB = BBList[f]

		img_data = list(img.getdata())
		#gt_data = list(gt.getdata())

		#If the bouding box is larger then 200x200, then resize the bb.
		if current_BB.size > 200*200:
			sizeRectangle = (0, 0)
			sizeRectangle[0] = math.fabs((BB[0][0] - BB[1][0]))
			sizeRectangle[1] = math.fabs((BB[0][1]) - BB[1][1])
			output = Image.new("RGB", sizeRectangle)
			output_data = list(output.getdata())
			for i, val in enumerate(img_data):
				current_position = convertLinear2Matrix(i, img.size[0], img.size[1])
				if current_position[0] >= current_BB.rectangle[0][0] and current_position[0] <= current_BB.rectangle[1][0] \
				and current_position[1] >= current_BB.rectangle[0][1] and current_position[1] <= current_BB.rectangle[1][1]:
					output_data[i] = val
				else:
					output_data[i] = (0, 0, 0)

			output.putdata(output_data)
			output = output.resize((200, 200), Image.ANTIALIAS)
			output.save(DATABASE +"output/"+ PPM_FILES[f][:-4]+"_resized" + PPM_FILES[f][-4:])
		else:	#Else, resize the original image
			img = img.resize((200, 200), Image.ANTIALIAS)
			img.save(DATABASE +"output/"+ PPM_FILES[f][:-4]+"_resized" + PPM_FILES[f][-4:])


if __name__ == "__main__":
	if not os.path.exists(DATABASE + "output/"):
		os.makedirs(DATABASE + "output/")
	main()








'''
img.putdata(img_data)

offset = (output.size[0]/2 - img.size[0]/2, output.size[1]/2 - img.size[1]/2)
output.paste(img,offset)
output = output.resize((200, 200), Image.ANTIALIAS)
output.save(DATABASE +"output/"+ PPM_FILES[f][:-4]+"_resized" + PPM_FILES[f][-4:])
'''

