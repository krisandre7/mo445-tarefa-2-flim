'''
Developed by: Azael de Melo e Sousa
04/07/2019
Adapted by: Taylla Milena Theodoro
20/03/2023
'''
import shutil
import os
import sys

import SimpleITK as sitk
from datetime import timedelta

def main():

	if (len(sys.argv) != 3):
		sys.exit("\nDependencies: iftNormalizeImage and iftInterp\n\nUsage: python %s <dicom dir> <output image>\nNumber of args: %d" % (sys.argv[0],len(sys.argv)))

	dirname = sys.argv[1]

	series_IDs = sitk.ImageSeriesReader_GetGDCMSeriesIDs(dirname)

	print("Converting Image")
	if not series_IDs:
		print('No series in directory ' + '\'' + dirname + '\'')

	series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dirname, series_IDs[0])
	series_reader = sitk.ImageSeriesReader()
	series_reader.SetFileNames(series_file_names)
	series_reader.MetaDataDictionaryArrayUpdateOn()
	series_reader.LoadPrivateTagsOn()

	image3D=series_reader.Execute()
	
	#####Based on DICOM standard
	
	#study_date_str  = series_reader.GetMetaData(0,"0008|0020");	
	study_time_str  = series_reader.GetMetaData(0,"0008|0030");
	
	##example of each time 
	#0008,0020,Study Date=20140412
	#0008,0021,Series Date=20140412
	#0008,0022,Acquisition Date=20140412
	#0008,0023,Image Date=20140412
	#0008,0030,Study Time=161237.368000
	#0008,0031, Series Time=161541.181000
	#0008,0032,Acquisition Time=161538.312500
	#0008,0033,Image Time=161541.183000 

	
	series_time_str = series_reader.GetMetaData(0,"0008|0031");
	
	#print(study_time_str)
	
	#print(series_time_str)
	#Calculate the time difference between them
	study_date  = timedelta(hours= int(study_time_str[:2]),
				minutes= int(study_time_str[2:4]), 
				seconds = int(study_time_str[4:6]))
	series_date = timedelta(hours= int(series_time_str[:2]),
				minutes= int(series_time_str[2:4]),
				seconds = int(series_time_str[4:6]))
	
	time_diff = series_date - study_date
	time_diff_str = str(int(time_diff.total_seconds()))
	
	img_filename = sys.argv[2]+ "_" + time_diff_str+ ".nii.gz"
	
	#for k in series_reader.GetMetaDataKeys(2):
	#	v = series_reader.GetMetaData(2,k)
	#	print("({0}) = = \"{1}\"".format(k,v))
	
	spacing = image3D.GetSpacing()

	sitk.WriteImage(image3D,img_filename)

	#for series in series_IDs:
	#	sitk.WriteImage(sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(dirname, series)),sys.argv[2])

	print("Done\n")

	create_tmp_file = False
	if (shutil.which("iftNormalizeImage") is None):
		print("iftNormalizeImage not compiled, skiping normalization.")
	else:
		create_tmp_file = True
		cmd = "iftNormalizeImage -i "+img_filename+" -a 0 -b 4095 -o temp.nii"
		print("Running command: "+cmd)
		if (os.system(cmd) == -1):
			print("Error at iftNormalizeImage. Please compile the program iftNormalizeImage")
			exit(-1)

	if (shutil.which("iftInterp") is None):
		print("iftInterp not compiled, skiping interpolation.")
	else:
		create_tmp_file = True
		minimum = min(spacing)
		cmd = "iftInterp temp.nii VOXEL %f %f %f %s" % (minimum, minimum, minimum, img_filename)
		print("Running command: "+cmd)
		if (os.system(cmd) == -1):
			print("Error at iftInterp. Please compile the program iftInterp")
			exit(-1)


	if (create_tmp_file):
		cmd = "rm temp.nii"
		if (os.system(cmd) == -1):
			print("Error removing temp")
			exit(-1)

if __name__ == "__main__":
	main()

