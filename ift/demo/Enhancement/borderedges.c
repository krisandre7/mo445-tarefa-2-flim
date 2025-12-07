#include "ift.h"
#include "iftTrainEdges.h"
#include "common.h"

int main(int argc, char **argv) 
{
	// read input files
	if (argc != 7)	iftError("Please provide the following parameters:\n <FILE_IMAGE_NAMES> <FILE_IMAGE_NAMES_GT> <SAMPLES_BORDER_BY_IMAGE> <PERC_BORDER> <RADIUS> <N_THREADS>\n\n", "main");
	char *file_imageNames;
	char *file_imageNamesGT;
        int nsamplesEdgesbi;
	float percBorder,radius;
        timer *tic, *toc;
        

	file_imageNames   = argv[1];
	file_imageNamesGT = argv[2];
        nsamplesEdgesbi   = atoi(argv[3]);
	percBorder        = atof(argv[4]);
	radius            = atof(argv[5]);

	//FIXME
	omp_set_num_threads(atoi(argv[6]));

        tic = iftTic();
	
	// Fill iftTrainEdges data
        iftTrainEdges *trainEdges;
	trainEdges = iftExtractTrainEdges(file_imageNames, file_imageNamesGT, nsamplesEdgesbi, percBorder, radius);
	
	// iftSaveTrainEdges();
	char *filenameTrainEdges;
	char *folder_name;
	folder_name = getFolderName(file_imageNames);
        char *auxpath = folder_name;
	filenameTrainEdges = strcat(auxpath,"TrainEdges");
	iftWriteTrainEdges(trainEdges,filenameTrainEdges);

	//// iftLoadTrainEdges();
	// iftTrainEdges *aux;
        // aux = iftReadTrainEdges(filenameTrainEdges);	
	// iftWriteTrainEdges(aux,filenameTrainEdges1);

        toc = iftToc();
	printf("Finished: %f\n", iftCompTime(tic,toc));
	return 0;
}
