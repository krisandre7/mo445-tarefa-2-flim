#include "ift.h"

typedef struct ift_SupDatasetProb {
   char *input_dir;
}  iftSupDatasetProb;

/*
//The 4 alphas weight: (mean band1, mean band2, mean band3), color histogram, LBP histogram, BIC histogram
float iftDistSupDescriptor(float *f1, float *f2, float *alpha, int n){
	float dist = 0;

	int i;
        
	for(i = 0; i < 3; i++){
		dist += powf(f1[i] - f2[i],2)*alpha[0];
	}
	
	for(i = 3; i < 3 + 216; i++){
		dist += powf(f1[i] - f2[i],2)*alpha[1];
	}
	
	for(i = 3 + 216; i < 3 + 216 + 256; i++){
		dist += powf(f1[i] - f2[i],2)*alpha[2];
	}
	
	for(i = 3 + 216 + 256; i < 3 + 3*216 + 256; i++){
		dist += powf(f1[i] - f2[i],2)*alpha[3];
	}
	
	return dist;
}
*/

//The 6 alphas weight: mean band1, mean band2, mean band3, color histogram, LBP histogram, BIC histogram
float iftDistSuperpixels(float *f1, float *f2, float *alpha, int n){
	float dist = 0;

	int i,bins_per_band,nbins;
        bins_per_band = (int)f1[n-1];
        nbins = bins_per_band*bins_per_band*bins_per_band;
	for(i = 0; i < 3; i++){
		dist += powf(f1[i] - f2[i],2)*alpha[i];
	}
	
	for(i = 3; i < 3 + nbins; i++){
		dist += powf(f1[i] - f2[i],2)*alpha[3];
	}
	
	for(i = 3 + nbins; i < 3 + nbins + 256; i++){
		dist += powf(f1[i] - f2[i],2)*alpha[4];
	}
	
	for(i = 3 + nbins + 256; i < 3 + 3*nbins + 256; i++){
		dist += powf(f1[i] - f2[i],2)*alpha[5];
	}
	
	return dist;
}


iftSupDatasetProb *iftCreateSupDatasetProb(char *input_dir)
{
   iftSupDatasetProb *prob=(iftSupDatasetProb *)calloc(1,sizeof(iftSupDatasetProb));
   prob->input_dir = input_dir;
   return(prob);
}

float iftEvalDataset(iftDataSet *Z, int nRUN, float percTrain, float cSVM) {

	int it;

	//printf("Num samples: %d, Num feats: %d, Num Class: %d \n",Z->nsamples,Z->nfeats,Z->nclasses);

	timer *tstart, *tfinish;
	//puts("");puts("");
	//printf("\ntraining and learning classification phase (cSVM:%.2f,nRuns:%d,pTrain:%6.2f)\n",cSVM,nRUN,100*percTrain);
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
		Ztmp = iftCopyDataSet(Z, true);

		tic = iftTic();
		iftSelectSupTrainSamples(Ztmp,percTrain); // Selecting training samples

		Ztrain[0] = iftExtractSamples(Ztmp,IFT_TRAIN);
		Ztest [0] = iftExtractSamples(Ztmp,IFT_TEST);
		iftDestroyDataSet(&Ztmp);

		Ztrain[1] = iftNormalizeDataSet(Ztrain[0]);
		iftDestroyDataSet(&Ztrain[0]);
		Ztest [1] = iftNormalizeTestDataSet(Ztrain[1],Ztest[0]);
		iftDestroyDataSet(&Ztest [0]);
		toc = iftToc();
		//fprintf(stderr,"selecting training(%d)/testing(%d) data: %f\n",Ztrain[1]->nsamples,Ztest[1]->nsamples,iftCompTime(tic,toc));

		tic = iftTic();
		if (cSVM > 0.0) {
			svm=iftCreateLinearSVC(cSVM);
			iftSVMTrainOVA(svm, Ztrain[1]); // Training
		}
		else if (cSVM < 0.0) {
		  float sigma = 0.1; //*Ztrain[1]->nfeats); // heuristically
		  svm = iftCreateRBFSVC(-cSVM, sigma);
		  iftSVMTrainOVO(svm,Ztrain[1]);
		}
		else {
		  graph = iftCreateCplGraph(Ztrain[1]);          // Training
		  iftSupTrain(graph);
		}
		toc = iftToc();
		//fprintf(stderr,"training svm classifier: %f\n", iftCompTime(tic,toc));


		tic = iftTic();
		if (cSVM > 0.0)
			iftSVMLinearClassifyOVA(svm, Ztest[1], Ztrain[1], IFT_TEST, NULL); // Classification
		else if (cSVM < 0.0)
			iftSVMClassifyOVO(svm, Ztest[1], IFT_TEST); // Classification
		else
			iftClassify(graph,Ztest[1]);                  // Classify test samples
		toc = iftToc();
		//fprintf(stderr,"classifying the test set: %f\n", iftCompTime(tic,toc));

	        acc[it] = iftSkewedTruePositives(Ztest[1]); // Compute accuracy on test set

		//printf("acc[%d] = %f\n",it,acc[it]);

		iftDestroyDataSet(&Ztrain[1]); // due to OPF usage, train data should be destroy only after test
		iftDestroyDataSet(&Ztest[1]);


		if (cSVM != 0.0)
			iftDestroySVM(svm);
		if (graph != NULL)
			iftDestroyCplGraph(&graph);
	}

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

	//fprintf(stdout,"Accuracy of classification is mean=%f, stdev=%f\n",mean,stdev);

	tfinish = iftToc();
	//printf("total classification by SVM phase: %f seg\n",iftCompTime(tstart,tfinish)/1000);
        return mean;
}


float iftSupDatasetMSPSFitness(void *prob, float *theta){
	int number_of_images,nRUN;
	iftImageNames  *image_names;
	float percTrain, result, cSVM;
	iftDataSet *Z;
	result = 0.0;
	nRUN = 30;
	percTrain = 0.5;
	cSVM = 1;
	iftSupDatasetProb *problem = (iftSupDatasetProb *) prob;
	number_of_images  = iftCountImageNames(problem->input_dir, "ds");
	image_names       = iftCreateAndLoadImageNames(number_of_images, problem->input_dir, "ds");
	//#pragma omp parallel for shared(number_of_images,image_names,input_dir)
	for (int s = 0; s < number_of_images ; s++){
		char filename[200];
		sprintf(filename,"%s/%s",problem->input_dir,image_names[s].image_name);
		//fprintf(stdout,"Processing %s\n",filename);
		Z  = iftReadOPFDataSet(filename);
		Z->iftArcWeight = iftDistSuperpixels;
		Z->function_number = IFT_NIL;
		result += iftEvalDataset(Z, nRUN, percTrain, cSVM);
		iftDestroyDataSet(&Z);
	}
	iftDestroyImageNames(image_names);
	result = result/ (float) number_of_images;
	return result;
}


void iftInitializeSupDatasetParam(iftMSPS *msps){
   
  /*Paramenters start with 1*/
  for (int i=0; i < 6; i++)
     msps->theta[i]=1;


  /*Parameters deltas*/
  /*
  msps->delta->val[iftGetMatrixIndex(msps->delta,0,0)] = 0.05;
  msps->delta->val[iftGetMatrixIndex(msps->delta,0,1)] = 0.1;
  msps->delta->val[iftGetMatrixIndex(msps->delta,0,2)] = 0.3;
  msps->delta->val[iftGetMatrixIndex(msps->delta,0,3)] = 0.7;
  */
  msps->delta->val[iftGetMatrixIndex(msps->delta,0,0)] = 0.5;
  msps->delta->val[iftGetMatrixIndex(msps->delta,0,1)] = 1;
  
  /*
  msps->delta->val[iftGetMatrixIndex(msps->delta,1,0)] = 0.05;
  msps->delta->val[iftGetMatrixIndex(msps->delta,1,1)] = 0.1;
  msps->delta->val[iftGetMatrixIndex(msps->delta,1,2)] = 0.3;
  msps->delta->val[iftGetMatrixIndex(msps->delta,1,3)] = 0.7;
  */
  msps->delta->val[iftGetMatrixIndex(msps->delta,1,0)] = 0.5;
  msps->delta->val[iftGetMatrixIndex(msps->delta,1,1)] = 1;

  /*
  msps->delta->val[iftGetMatrixIndex(msps->delta,2,0)] = 0.05;
  msps->delta->val[iftGetMatrixIndex(msps->delta,2,1)] = 0.1;
  msps->delta->val[iftGetMatrixIndex(msps->delta,2,2)] = 0.3;
  msps->delta->val[iftGetMatrixIndex(msps->delta,2,3)] = 0.7;
  */
  msps->delta->val[iftGetMatrixIndex(msps->delta,2,0)] = 0.5;
  msps->delta->val[iftGetMatrixIndex(msps->delta,2,1)] = 1;
   
  /*
  msps->delta->val[iftGetMatrixIndex(msps->delta,3,0)] = 0.05;
  msps->delta->val[iftGetMatrixIndex(msps->delta,3,1)] = 0.1;
  msps->delta->val[iftGetMatrixIndex(msps->delta,3,2)] = 0.3;
  msps->delta->val[iftGetMatrixIndex(msps->delta,3,3)] = 0.7;
  */
  msps->delta->val[iftGetMatrixIndex(msps->delta,3,0)] = 0.5;
  msps->delta->val[iftGetMatrixIndex(msps->delta,3,1)] = 1;

  msps->delta->val[iftGetMatrixIndex(msps->delta,4,0)] = 0.5;
  msps->delta->val[iftGetMatrixIndex(msps->delta,4,1)] = 1;

  msps->delta->val[iftGetMatrixIndex(msps->delta,5,0)] = 0.5;
  msps->delta->val[iftGetMatrixIndex(msps->delta,5,1)] = 1;
  
}



int main(int argc, char *argv[]) 
{
  char *input_dir;
  float result;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc!=2)
    iftError("Usage: iftMSPSDataset <dataset_directory>","main");

  input_dir = argv[1];
  iftMSPS *msps;
  iftSupDatasetProb *prob;
  prob = iftCreateSupDatasetProb(input_dir);
  msps = iftCreateMSPS(6, 2, iftSupDatasetMSPSFitness, prob);
  msps->niters = 10000;
  //printf("niters: %d\n",msps->niters);
  iftInitializeSupDatasetParam(msps);
  result = iftMSPSMax(msps);
  printf("Final: %f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f\n",result, msps->theta[0], msps->theta[1], msps->theta[2], msps->theta[3], msps->theta[4], msps->theta[5]);
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}

