#!/usr/bin/python2.6
import os
import sys
import re
import time
import random

def copyfile(source, dest, buffer_size=1024*1024):
    """
    Copy a file from source to dest. source and dest
    can either be strings or any object with a read or
    write method, like StringIO for example.
    """
    if not hasattr(source, 'read'):
        source = open(source, 'rb')
    if not hasattr(dest, 'write'):
        dest = open(dest, 'wb')

    while 1:
        copy_buffer = source.read(buffer_size)
        if copy_buffer:
            dest.write(copy_buffer)
        else:
            break

    source.close()
    dest.close()

def main ():	
	if len(sys.argv) != 4:
		print "Error. Please provide:"
		print "<database> <databaseGT> <number_images>"
		return 1
	# read input
	database = sys.argv[1]
        databaseGT = sys.argv[2]
	number_images = int(sys.argv[3])
	currentDir = ""
	if database[0]!='/':
		currentDir = os.path.dirname(os.path.abspath(__file__))+"/"
	destFolder = database+"-train"
	destFolderGT = databaseGT+"-train"
	# remove dest
	cmd = "rm -rf "+destFolder
	os.system(cmd)
	cmd = "rm -rf "+destFolderGT
	os.system(cmd)
	# mkdir dest 
	cmd = "mkdir "+destFolder
	os.system(cmd)
	cmd = "mkdir "+destFolderGT
	os.system(cmd)
	# get files
	output = []
	outputGT = []
	files = [ f for f in os.listdir(database) if f.endswith(".pgm")]
        filesGT = [ f for f in os.listdir(databaseGT) if f.endswith(".pgm")]
	# random order
	random.shuffle(files)
	# select top number_images
        selectedFiles = files[:number_images]
	# sort selected images
	selectedFiles.sort()
        # select GT images
	selectedFilesGT = []
        j = 0 
        for e in selectedFiles:		
   		el  = [ f for f in filesGT if f.startswith(e[:-4])];
		selectedFilesGT.append(el[0])
		j=j+1
	# list selected images
	j=0
	for f in selectedFiles:
		#output.append(destFolder+"/"+f)
		#outputGT.append(destFolderGT+"/"+selectedFilesGT[j])
		output.append(f)
		outputGT.append(selectedFilesGT[j])
		# set source and dest
		source = currentDir+database+"/"+f
		dest = currentDir+destFolder+"/"+f
                sourceGT = currentDir+databaseGT+"/"+selectedFilesGT[j]
		destGT = currentDir+destFolderGT+"/"+selectedFilesGT[j]
		# copy file
		#copyfile(source,dest)
                # create symbolic links
		cmd = "ln -s "+source+" "+dest
		os.system(cmd)
		cmd = "ln -s "+sourceGT+" "+destGT
		os.system(cmd)
		j=j+1
	# write list od selected images
	output_fp = open(destFolder+"/filenames.txt", "w")
	output_fp.write("\n".join(output))
	output_fp.close()
	output_fp = open(destFolderGT+"/filenames.txt", "w")
	output_fp.write("\n".join(outputGT))
	output_fp.close()
	print "Files saved in: "
	print "\""+destFolder+"\""
	print " and "
	print "\""+destFolderGT+"\""

main ()
