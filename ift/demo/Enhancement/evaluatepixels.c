#include "common.h"
#include "ift.h"
#include "iftTrainPixels.h"

int main(int argc, char **argv) 
{
	if (argc != 7)	iftError("Please provide the following parameters:\n<TRAIN_PIXELS_FILE> <DATASET_PIXELS_FILE> <NRUN> <PERC_TRAIN> <cSVM(0=OPF)> <N_THREADS>\n\n", "main");


	int idxImg,totalPixels;
	int it,nRUN;
	float cSVM,percTrain;
	char *file_trainpixels;
	char *file_datasetpixels;

	file_trainpixels   = argv[1];
	file_datasetpixels = argv[2];
	nRUN      = atoi(argv[3]);
	percTrain = atof(argv[4]);
	cSVM      = atof(argv[5]);
	omp_set_num_threads(atoi(argv[6]));

	printf("Train pixels file: %s\n", file_trainpixels);
	printf("DataSet pixels file: %s\n", file_datasetpixels);	
	puts("");

	printf("Reading TrainPixels file and corresponding input images\n");
	iftTrainPixels* trainpixels = iftReadTrainPixels(file_trainpixels,1);
	printf("Number of images read: %d\n", trainpixels->nimages);
	
	for(totalPixels = 0,idxImg = 0; idxImg < trainpixels->nimages ; idxImg++)
		totalPixels += trainpixels->data[idxImg].n;
	printf("%d pixels to be processed\n",totalPixels);

	printf("Reading TrainPixels file and corresponding input images\n");
	iftDataSet *Zpixels = iftReadOPFDataSet(file_datasetpixels);
	
	if (totalPixels != Zpixels->nsamples)	{
	  char msg[100];
	  sprintf(msg,"Number of pixels from TrainPixels (%d) and DataSetPixels (%d) are different\n\n",totalPixels,Zpixels->nsamples);
	  iftError(msg, "main");
	}
	
	timer *tstart, *tfinish;
	puts("");puts("");	
	printf("\ntraining and learning classification phase (cSVM:%.2f,nRuns:%d,pTrain:%6.2f)\n",cSVM,nRUN,100*percTrain);
	tstart = iftTic();

	float* acc  = iftAllocFloatArray(nRUN); 

#pragma omp parallel for shared(acc)
	for(it=0;it<nRUN;it++)
	{
		timer *tic,*toc;
		iftSVM      *svm   = NULL;
		iftCplGraph *graph = NULL;
		iftDataSet *Ztrain[2], *Ztest[2],*Ztmp;

		// due to paralelism a copy of the dataset is required for choosing samples
		Ztmp = iftCopyDataSet(Zpixels);

		tic = iftTic();
		//iftSelectSupTrainSamples(Ztmp,percTrain); // Selecting training samples
		// Selecting training images - leave pixels out of %(1-percTrain) images
		iftSelectSupTrainPixels(Ztmp,trainpixels,percTrain,0.5);
		// 60% of border pixels for OPF training, empirically chosen

		Ztrain[0] = iftExtractSamples(Ztmp,IFT_TRAIN);
		Ztest [0] = iftExtractSamples(Ztmp,IFT_TEST);
		iftDestroyDataSet(&Ztmp);

		Ztrain[1] = iftNormalizeDataSet(Ztrain[0]);
		iftDestroyDataSet(&Ztrain[0]);
		Ztest [1] = iftNormalizeTestDataSet(Ztrain[1],Ztest[0]);
		iftDestroyDataSet(&Ztest [0]);
		toc = iftToc();
		fprintf(stderr,"selecting training(%d)/testing(%d) data: %f\n",Ztrain[1]->nsamples,Ztest[1]->nsamples,iftCompTime(tic,toc));
		
		tic = iftTic();	
		if (cSVM > 0.0) {
			svm=iftCreateLinearSVC(cSVM);
			iftSVMTrainOVA(svm, Ztrain[1]); // Training
		}
		else if (cSVM < 0.0) {
		  float sigma = 0.1; //*Ztrain[1]->nfeats); // heuristically
		  svm = iftCreateRBFSVC(-cSVM, sigma);
		  iftSVMTrainOVA(svm,Ztrain[1]);
		}
		else {
		  graph = iftCreateCplGraph(Ztrain[1]);          // Training
		  iftSupTrain(graph);
		}
		toc = iftToc();
		fprintf(stderr,"training svm classifier: %f\n", iftCompTime(tic,toc));

	
		tic = iftTic();
		if (cSVM > 0.0)
			iftSVMLinearClassifyOVA(svm, Ztest[1], IFT_TEST); // Classification
		else if (cSVM < 0.0)
			iftSVMClassifyOVA(svm, Ztest[1], IFT_TEST); // Classification
		else
			iftClassify(graph,Ztest[1]);                  // Classify test samples
		toc = iftToc();
		fprintf(stderr,"classifying the test set: %f\n", iftCompTime(tic,toc));

		acc[it] = iftSkewedTruePositives(Ztest[1]); // Compute accuracy on test set
		printf("acc[%d] = %f\n",it,acc[it]);

		iftDestroyDataSet(&Ztrain[1]); // due to OPF usage, train data should be destroy only after test
		iftDestroyDataSet(&Ztest[1]);    


		if (cSVM != 0.0)
			iftDestroySVM(svm);
		if (graph != NULL)
			iftDestroyCplGraph(&graph);
	}

	iftDestroyDataSet(&Zpixels);

	float mean = 0.0;
	for (it=0; it < nRUN; it++)
		mean += acc[it];
	mean /= nRUN;

	float stdev = 0.0;
	for (it=0; it < nRUN; it++)
	  stdev += (acc[it]-mean)*(acc[it]-mean);
	if (nRUN > 1)
	  stdev = sqrtf(stdev/(nRUN-1));

	free(acc); 

	fprintf(stdout,"Accuracy of classification is mean=%f, stdev=%f\n",mean,stdev); 
	
	tfinish = iftToc();
	printf("total classification by SVM phase: %f seg\n",iftCompTime(tstart,tfinish)/1000);

	return 0;
}
