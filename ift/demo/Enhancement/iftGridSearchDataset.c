#include "ift.h"

float* iftGridSearch(iftDataSet* Z, float percTrainVal, int nRUN){

	iftDataSet *Ztrain[2], *Ztest[2],*Ztmp;
	iftSVM      *svm   = NULL;
	int i,j,it;
	float C, sigma;
	float max_acc = 0;
	float *values;
	values = iftAllocFloatArray(2);
	float sigmas[5], Cs[5];
	sigmas[0] = 0.010000;
	sigmas[1] = 0.056234;
	sigmas[2] = 0.316228;
	sigmas[3] = 1.778279;
	sigmas[4] = 10.000000;
	Cs[0] = 1;
	Cs[1] = 10;
	Cs[2] = 100;
	Cs[3] = 1000;
	Cs[4] = 10000;
	C = 100000;
	sigma = 0.1;
	for(i=0; i<5; i++){
		for(j=0; j<5; j++){
			float acc = 0;
			for(it=0;it<nRUN;it++){
				Ztmp = iftCopyDataSet(Z, true);
				iftSetStatus(Ztmp, IFT_TEST);
				iftSelectSupTrainSamples(Ztmp,percTrainVal); // Selecting training samples
				/* Normalize Train and Test */
				Ztrain[0] = iftExtractSamples(Ztmp, IFT_TRAIN);
				Ztest[0] = iftExtractSamples(Ztmp, IFT_TEST);

				Ztrain[1] = iftNormalizeDataSet(Ztrain[0]);
				iftDestroyDataSet(&Ztrain[0]);
				Ztest [1] = iftNormalizeTestDataSet(Ztrain[1],Ztest[0]);
				iftDestroyDataSet(&Ztest [0]);

				/* Train RBF Kernel */
				svm = iftCreateRBFSVC(Cs[i], sigmas[j]);
				iftSVMTrainOVO(svm,Ztrain[1]);

				/* Classify RBF Kernel */
				iftSVMClassifyOVO(svm, Ztest[1], IFT_TEST); // Classification
				acc += iftTruePositives(Ztest[1]) / (float) nRUN;

				iftDestroyDataSet(&Ztrain[1]);
				iftDestroyDataSet(&Ztest[1]);

				iftDestroySVM(svm);
				iftDestroyDataSet(&Ztmp);

			}
			printf("s=%f, C=%f mean acc: %f\n", Cs[i], sigmas[j], acc);
			if(acc > max_acc){
				max_acc = acc;
				C = Cs[i];
				sigma = sigmas[j];
			}
		}
	}
	values[0] = C;
	values[1] = sigma;
	return values;
}


int main(int argc, char **argv) 
{
	if (argc != 4)	iftError("Please provide the following parameters:\n<DATASET_FILE> <NRUN> <PERC_TRAIN>\n\n", "main");

	int nRUN;
	float percTrain, percTrainVal, accuracy;
	char *file_dataset;
	iftSVM      *svm   = NULL;

	file_dataset = argv[1];
	nRUN      = atoi(argv[2]);
	percTrain = atof(argv[3]);
	percTrainVal = 0.33;
	omp_set_num_threads(6);

	srand(time(NULL));

	printf("Reading Dataset file and corresponding input images\n");
	iftDataSet *Z = iftReadOPFDataSet(file_dataset);
	printf("Num samples: %d, Num feats: %d, Num Class: %d \n",Z->nsamples,Z->nfeats,Z->nclasses);

	//timer *tstart, *tfinish;


	iftDataSet *Ztrain[2], *Ztest[2], *Train, *Test;


	//timer *tic,*toc;
	iftSetStatus(Z,IFT_TRAIN);
	iftSelectSupTrainSamples(Z,percTrain); // Selecting training samples
	printf("nsamples: %d, ntrainsamples: %d\n", Z->nsamples, Z->ntrainsamples);
	Train = iftExtractSamples(Z, IFT_TRAIN);
	Test = iftExtractSamples(Z, IFT_TEST);
	float C, sigma;
	float *valuesGrid;
    valuesGrid = iftAllocFloatArray(2);

	valuesGrid = iftGridSearch(Train, percTrainVal, nRUN);
	C = valuesGrid[0];
	sigma = valuesGrid[1];
	//free(valuesGrid);
	printf("C : %f, sigma: %f \n", C, sigma);

	/* Normalize Train and Test */
	Ztrain[1] = iftNormalizeDataSet(Train);
	iftDestroyDataSet(&Train);
	Ztest [1] = iftNormalizeTestDataSet(Ztrain[1],Test);
	iftDestroyDataSet(&Test);

	/* Train RBF Kernel */
	svm = iftCreateRBFSVC(C, sigma);
	iftSVMTrainOVO(svm,Ztrain[1]);

	/* Classify RBF Kernel */
	iftSVMClassifyOVO(svm, Ztest[1], IFT_TEST); // Classification
	accuracy = iftTruePositives(Ztest[1]);

	printf("Accuracy %f\n",accuracy);

	iftDestroyDataSet(&Ztrain[1]);
	iftDestroyDataSet(&Ztest[1]);

	iftDestroySVM(svm);

	return 0;
}
