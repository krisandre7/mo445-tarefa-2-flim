#!/usr/bin/python2.7
import sys
import os, os.path
import argparse
import numpy as np

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("origDir", type=str, help="Image directory")
	parser.add_argument("labelDir", type=str, help="Label directory")	
	parser.add_argument("outputDir", type=str, help="Output directory")
	parser.add_argument("numTrain", type=int, help="Number of random train samples")
	args = parser.parse_args()
	
	origDir = args.origDir
	labelDir = args.labelDir
	outputDir = args.outputDir
	numTrain = args.numTrain
	
	origDir = os.path.abspath(origDir)
	labelDir = os.path.abspath(labelDir)
	outputDir = os.path.abspath(outputDir)
	outTrainOrigDir = "{}/{}/{}".format(outputDir, "train", "orig")
	outTrainLabelDir = "{}/{}/{}".format(outputDir, "train", "label")
	outTestOrigDir = "{}/{}/{}".format(outputDir, "test", "orig")
	outTestLabelDir = "{}/{}/{}".format(outputDir, "test", "label")
	
	if os.path.isdir(outTrainOrigDir) or os.path.isdir(outTrainLabelDir):
		print("Files exist in {} ".format(outTrainOrigDir))
		return 0
	os.makedirs(outTrainOrigDir)
	os.makedirs(outTrainLabelDir)
	os.makedirs(outTestOrigDir)
	os.makedirs(outTestLabelDir)
	
	origFiles = [ f for f in os.listdir(origDir) if os.path.isfile(os.path.join(origDir,f)) ]
	labelFiles = [f for f in os.listdir(labelDir) if os.path.isfile(os.path.join(labelDir,f)) ]	
	origFiles.sort()
	labelFiles.sort()
	
	origFiles = np.array(origFiles)
	labelFiles = np.array(labelFiles)
	numSamples = len(origFiles)

	# Select training samples
	randomList = np.arange(numSamples)
	np.random.shuffle(randomList)
	selectedTrainRandom = randomList[:numTrain]
	selectedTestRandom = randomList[numTrain:]
	
	origTrainFiles = origFiles[selectedTrainRandom]
	labelTrainFiles = labelFiles[selectedTrainRandom]
	origTestFiles = origFiles[selectedTestRandom]
	labelTestFiles = labelFiles[selectedTestRandom]
	
	# write train files
	for origFile in origTrainFiles:
		os.symlink(os.path.join(origDir,origFile), os.path.join(outTrainOrigDir,origFile))
	for labelFile in labelTrainFiles:
		os.symlink(os.path.join(labelDir,labelFile), os.path.join(outTrainLabelDir,labelFile))
	# write test files
	for origFile in origTestFiles:
		os.symlink(os.path.join(origDir,origFile), os.path.join(outTestOrigDir,origFile))
	for labelFile in labelTestFiles:
		os.symlink(os.path.join(labelDir,labelFile), os.path.join(outTestLabelDir,labelFile))
	
	
if __name__ == '__main__':
    main()
