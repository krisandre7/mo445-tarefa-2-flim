#include "ift.h"


int main(int argc, char **argv) 
{
	if (argc != 9)	iftError("Please provide the following parameters:\n<FEATS_FILE CSV> <rows1> <columns1> <CLASS_FILE CSV> <rows2> <columns2> <DATASET> <nclasses>\n\n", "main");

	char *filename_feats, *filename_class, *filename_dataset;
	int rows1, cols1, rows2, cols2;
	int nclasses;

	filename_feats = argv[1];
	rows1 = atoi(argv[2]);
	cols1 = atoi(argv[3]);
	filename_class = argv[4];
	rows2 = atoi(argv[5]);
	cols2 = atoi(argv[6]);
	filename_dataset = argv[7];
	nclasses = atoi(argv[8]);

	printf("Reading CSV files\n");
	iftDataSet *Z;

	FILE *f1 = fopen(filename_feats, "r");
	FILE *f2 = fopen(filename_class, "r");
	
	int i,j;
	float temp;
        float *values1,*values2;
        values1 = iftAllocFloatArray(rows1*cols1);
	values2 = iftAllocFloatArray(rows2*cols2);
	for(i=0; i<rows1; i++){
		for(j=0; j<cols1; j++){
			fscanf(f1,"%f",&temp);
			values1[i*cols1+j] = temp;
			fscanf(f1,",");
		}
	}
	fclose(f1);

	for(i=0; i<rows2; i++){
		for(j=0; j<cols2; j++){
			fscanf(f2,"%f",&temp);
			values2[i*cols2+j] = temp;
		}
	}
	fclose(f2);
	
	/* Fill Dataset */
	Z = iftCreateDataSet(rows1,cols1);
	Z->nsamples = rows1;
	Z->nclasses = nclasses;
	Z->nfeats = cols1;
	for(i=0; i<rows1; i++){
		Z->sample[i].class = (int)values2[i];
		for(j=0; j<cols1; j++){
			Z->sample[i].feat[j] = values1[i*cols1+j];
		}
	}
	
	iftWriteOPFDataSet(Z, filename_dataset);
	
	free(values1);
	free(values2);
	
        iftDestroyDataSet(&Z);
	return 0;
}
