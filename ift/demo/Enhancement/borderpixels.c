#include "ift.h"
#include "iftTrainPixels.h"
#include "common.h"

int main(int argc, char **argv) 
{
	// read input files
	if (argc != 6)	iftError("Please provide the following parameters:\n <FILE_IMAGE_NAMES> <FILE_IMAGE_NAMES_GT> <SAMPLES_BY_IMAGE> <PERC_BORDER> <RADIUS>\n\n", "main");
	char *file_imageNames;
	char *file_imageNamesGT;
	int nsamplesbi;
	float percBorder,radius;
        timer *tic, *toc;

	file_imageNames   = argv[1];
	file_imageNamesGT = argv[2];
        nsamplesbi        = atoi(argv[3]);
	percBorder        = atof(argv[4]);
	radius            = atof(argv[5]);

        tic = iftTic();
	
	// Fill iftTrainEdges data
        iftTrainPixels *trainPixels = iftExtractTrainPixels(file_imageNames, file_imageNamesGT, nsamplesbi, percBorder, radius);
	
	// iftSaveTrainEdges();
	char *filenameTrainPixels = iftAllocCharArray(256);
	char *folder_name;
	folder_name = getFolderName(file_imageNames);
	sprintf(filenameTrainPixels,"%sTrainPixels.n%03d.sbI%05d.pB%03d.r%03d.bin",folder_name,trainPixels->nimages,nsamplesbi,(int)(1000*percBorder),(int)(10*radius));
	
	iftWriteTrainPixels(trainPixels,filenameTrainPixels);

//	iftTrainPixels *aux = iftReadTrainPixels(filenameTrainPixels,1);	
//	iftWriteTrainPixels(aux,filenameTrainPixels);

        toc = iftToc();
	printf("Finished: %f\n", iftCompTime(tic,toc));
	
	iftDestroyTrainPixels(&trainPixels);
	free(filenameTrainPixels);

	return 0;
}
