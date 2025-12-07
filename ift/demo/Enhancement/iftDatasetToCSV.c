#include "ift.h"


int main(int argc, char **argv) 
{
	if (argc != 4)	iftError("Please provide the following parameters:\n<DATASET_FILE> <FEATS_DATASET> <CLASS_DATASET> \n\n", "main");

	char *file_dataset;

	file_dataset = argv[1];

	printf("Reading Dataset file and corresponding input images\n");
	iftDataSet *Z = iftReadOPFDataSet(file_dataset);
	printf("Num samples: %d, Num feats: %d, Num Class: %d \n",Z->nsamples,Z->nfeats,Z->nclasses);

	FILE *f1 = fopen(argv[2], "w");
	FILE *f2 = fopen(argv[3], "w");

	int i,j;
	for (i = 0; i < Z->nsamples; i++) {
		for (j=0; j < Z->nfeats; j++) {
			fprintf(f1,"%.4f", Z->sample[i].feat[j]);
			if(j == (Z->nfeats-1))
				fprintf(f1,"\n");
			else
				fprintf(f1,",");
		}
	}
	for (i = 0; i < Z->nsamples; i++) {
		fprintf(f2,"%d\n", Z->sample[i].class);
	}

	fclose(f1);
	fclose(f2);
	
        iftDestroyDataSet(&Z);
	return 0;
}
