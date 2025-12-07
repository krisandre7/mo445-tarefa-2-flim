#include "common.h"
#include "ift.h"
#include "iftTrainEdges.h"

#include <libgen.h>

int  iftAddLabeledEdgesToDataSetFromMImage(iftMImage* mimg, iftLabelEdgesData* edgeData,iftDataSet* Zedges,int offsetSample,int offsetID,iftAdjRel* A)
{
	fprintf(stderr,"edges: %d, pixelsize: %d, dataset size: %d\n",(A->n-1)*mimg->n,mimg->m,Zedges->nsamples);

	// collecting edges from trainedges to Z
	int s,sA,b,p,q,class;
	for(s = 0,sA = 0; s < edgeData->n; s++)
	{
		p     = edgeData->labelEdge[s].p;
		q     = edgeData->labelEdge[s].q;
		class = edgeData->labelEdge[s].truelabel;
			
		Zedges->sample[offsetSample+s].truelabel = class;
		Zedges->sample[offsetSample+s].id    = offsetID + p *(A->n-1) + q;
		
		for(b = 0; b < mimg->m ; b++)
			Zedges->sample[offsetSample+s].feat[b] = fabs(mimg->val[p][b] - mimg->val[q][b]);
		sA++;
	}

	return sA;	
}

//And here it begins....
int main(int argc, char **argv) 
{
	if (argc != 4)	iftError("Please provide the following parameters:\n<CONFIG_FILE> <TRAIN_EDGES_FILE> <N_THREADS>\n\n", "main");

 	int idxImg,offsetSample=0,offsetID=0,it,scale;
	char *file_msconvnet  ; // = iftAllocCharArray(64);
	char *file_trainedges ; // = iftAllocCharArray(64);

	//FIXME
	omp_set_num_threads(atoi(argv[3]));
 
	file_msconvnet   = argv[1];
	file_trainedges  = argv[2];

	printf("Multi-scale convolution network configuration file: %s\n"   , file_msconvnet);
	printf("Train edges file: %s\n", file_trainedges);
	puts("");


	//FIXME
	iftMSConvNetwork* msconvnet = parseConfigFile(file_msconvnet);
	int sizeFeatures = 0;
	for(scale = 0; scale < msconvnet->nscales; scale++)
//		sizeFeatures += (msconvnet->k_bank[msconvnet->nlayers-1])->K[0]->nbands;
		sizeFeatures += ((msconvnet->convnet[scale])->k_bank[msconvnet->convnet[scale]->nlayers-1])->nkernels;
	printf("feature size: %d\n",sizeFeatures);

	//FIXME	
	printf("Reading TrainEdges file and corresponding input images\n");
	iftTrainEdges* trainedges = iftReadTrainEdges(file_trainedges);
	printf("Number of images read: %d\n", trainedges->nimages);
	
//	char* dirname_trainedges  = dirname(file_trainedges);
//	char* basename_trainedges = basename(file_trainedges);
	
	int totalEdges = 0.;
	for(idxImg = 0; idxImg < trainedges->nimages ; idxImg++)
		totalEdges += trainedges->data[idxImg].n;
	printf("%d edges to be processed\n",totalEdges);
  
	iftAdjRel* A;
//	if (mimg->zsize == 1) A = iftCircularEdges(trainedges->radius);
//	else                  A = iftSphericEdges(trainedges->radius);
	A = iftCircularEdges(trainedges->radius);

	iftDataSet *Zedges;
	
	Zedges = iftCreateDataSet(totalEdges, sizeFeatures);
	Zedges->nclasses = 2;
	
	timer *tic, *toc,*t1, *t2, *tstart, *tfinish;
	
	printf("\nfeature learning process started with msconvnet parameters\n");
	//	#pragma omp parallel for private(fullPath)
	tstart = iftTic();	
	for(idxImg = 0; idxImg < trainedges->nimages ; idxImg++)
	{
		t1 = iftTic();
		
//		sprintf("%s/%s",dirname,trainedges->data[idxImg].filename);
		puts("");		
		printf("Processing: (%d/%d) %s\n", idxImg+1,trainedges->nimages,trainedges->data[idxImg].filename);

		// Generating Deep Features and Edges
		tic = iftTic();
		iftMImage*  mimg  = iftApplyMSConvNetwork(trainedges->data[idxImg].image, msconvnet);
		toc = iftToc();
		printf("Feature learning: %f\n", iftCompTime(tic,toc));


		// Creating training SET
		tic = iftTic();
		offsetSample = (idxImg == 0 ? 0 : offsetSample + trainedges->data[idxImg].n);
		offsetID     = (idxImg == 0 ? 0 : offsetID     + trainedges->data[idxImg].image->n);
		int nedges   = iftAddLabeledEdgesToDataSetFromMImage(mimg,&(trainedges->data[idxImg]),Zedges,offsetSample,offsetID,A);
		toc = iftToc();
		printf("Adding %d edges to dataset: %f\n", nedges, iftCompTime(tic,toc));

		iftDestroyMImage(&mimg);

		t2 = iftToc();
		printf("feature extraction: %f seg\n",iftCompTime(t1,t2)/1000);
	}
	iftDestroyAdjRel(&A);

	fprintf(stderr,"destroying MSConvNetwork...\n");
	iftDestroyMSConvNetwork(&msconvnet);

	fprintf(stderr,"destroying TrainEdges...\n");
	iftDestroyTrainEdges(&trainedges);

	tfinish = iftToc();
	printf("total feature extraction: %f seg\n",iftCompTime(tstart,tfinish)/1000);


	// Training/testing
	puts("");puts("");	
	printf("\ntraining and learning classification phase\n");
	tstart = iftTic();

	iftDataSet *Ztrain[2], *Ztest[2];
	int n = 30;  
	float* acc  = iftAllocFloatArray(n); 

	for(it=0;it<n;it++)
	{
		tic = iftTic();	
		iftSelectSupTrainSamples(Zedges,0.5); // Selecting training samples

		Ztrain[0] = iftExtractSamples(Zedges,IFT_TRAIN);
		Ztest [0] = iftExtractSamples(Zedges,IFT_TEST);

		Ztrain[1] = iftNormalizeDataSet(Ztrain[0]);
		iftDestroyDataSet(&Ztrain[0]);
		Ztest [1] = iftNormalizeTestDataSet(Ztrain[1],Ztest[0]);
		iftDestroyDataSet(&Ztest [0]);
		toc = iftToc();
		fprintf(stderr,"selecting training/testing data: %f\n", iftCompTime(tic,toc));
		
		tic = iftTic();	
		float C = 1e5;
//		float C = 1.;
		iftSVM *svm=iftCreateLinearSVC(C);
		iftSVMTrainOVA(svm, Ztrain[1]); // Training
		iftDestroyDataSet(&Ztrain[1]);
		toc = iftToc();
		fprintf(stderr,"training svm classifier: %f\n", iftCompTime(tic,toc));

	
		tic = iftTic();
		iftSVMLinearClassifyOVA(svm, Ztest[1], IFT_TEST); // Classification
		toc = iftToc();
		fprintf(stderr,"classifying the test set: %f\n", iftCompTime(tic,toc));

		acc[it] = iftSkewedTruePositives(Ztest[1]); // Compute accuracy on test set
		iftDestroyDataSet(&Ztest[1]);    
		printf("acc[%d] = %f\n",it,acc[it]);

		iftDestroySVM(svm);
	}

	float mean = 0.0;
	for (it=0; it < n; it++)
		mean += acc[it];
	mean /= n;

	float stdev = 0.0;
	for (it=0; it < n; it++)
	  stdev += (acc[it]-mean)*(acc[it]-mean);
	if (n > 1)
	  stdev = sqrtf(stdev/(n-1));

	free(acc); 

	fprintf(stdout,"Accuracy of classification is mean=%f, stdev=%f\n",mean,stdev); 
	
	tfinish = iftToc();
	printf("total classification by SVM phase: %f seg\n",iftCompTime(tstart,tfinish)/1000);

	return 0;
}