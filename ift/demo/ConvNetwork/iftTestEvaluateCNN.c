#include "ift.h"


float iftSensitivityMulticlass(iftDataSet *Z, int class){
  int     i, TP=0, FN=0;
  float   sensitivity;

  for (i = 0; i < Z->nsamples; i++){
    if (Z->sample[i].status != IFT_TRAIN){
      if(Z->sample[i].truelabel == class){
        if(Z->sample[i].truelabel == Z->sample[i].label){
	  TP++;
        }else{
      FN++;
        }
      }
    }
  }
  sensitivity = (float)TP / ((float)TP + (float)FN);

  return(sensitivity);
}

float iftSpecificityMulticlass(iftDataSet *Z, int class){
  int     i, TN=0, FP=0;
  float   specificity;

  for (i = 0; i < Z->nsamples; i++){
    if (Z->sample[i].status != IFT_TRAIN){
      if(Z->sample[i].truelabel != class){
        if(Z->sample[i].label != class){
	  TN++;
        }else{
      FP++;
        }
      }
    }
  }
  specificity = (float)TN / ((float)TN + (float)FP);

  return(specificity);
}

void iftEvaluateCNN(iftDataSet *Z, int nRUN, float percTrain, float cSVM)
{

	int idxImg,totalPixels;
	int it,i;

	printf("Num samples: %d, Num feats: %d, Num Class: %d \n",Z->nsamples,Z->nfeats,Z->nclasses);

	timer *tstart, *tfinish;
	puts("");puts("");
	printf("\ntraining and learning classification phase (cSVM:%.2f,nRuns:%d,pTrain:%6.2f)\n",cSVM,nRUN,100*percTrain);
	tstart = iftTic();

	int nclass = Z->nclasses;
	float* acc  = iftAllocFloatArray(nRUN);
	float* sensv  = iftAllocFloatArray(nRUN*nclass);
	float* specf  = iftAllocFloatArray(nRUN*nclass);

#pragma omp parallel for shared(acc,sensv,specf)
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
		fprintf(stderr,"selecting training(%d)/testing(%d) data: %f\n",Ztrain[1]->nsamples,Ztest[1]->nsamples,iftCompTime(tic,toc));

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
		fprintf(stderr,"training svm classifier: %f\n", iftCompTime(tic,toc));


		tic = iftTic();
		if (cSVM > 0.0)
			iftSVMLinearClassifyOVA(svm, Ztest[1], Ztrain[1], IFT_TEST, NULL); // Classification
		else if (cSVM < 0.0)
			iftSVMClassifyOVO(svm, Ztest[1], IFT_TEST); // Classification
		else
			iftClassify(graph,Ztest[1]);                  // Classify test samples
		toc = iftToc();
		fprintf(stderr,"classifying the test set: %f\n", iftCompTime(tic,toc));

        acc[it] = iftSkewedTruePositives(Ztest[1]); // Compute accuracy on test set

        for(i=0;i<nclass;i++){
          sensv[it*nclass+i] = iftSensitivityMulticlass(Ztest[1], i+1);
          specf[it*nclass+i] = iftSpecificityMulticlass(Ztest[1], i+1);
        }

		printf("acc[%d] = %f\n",it,acc[it]);

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

	float mean_sensv_all = 0.0;
	float mean_specf_all = 0.0;
	for(i=0;i<nclass;i++){

		float mean_sensv = 0.0;
		for (it=0; it < nRUN; it++)
			mean_sensv += sensv[it*nclass+i];
		mean_sensv /= nRUN;

		float stdev_sensv = 0.0;
		for (it=0; it < nRUN; it++)
			stdev_sensv += (sensv[it*nclass+i]-mean_sensv)*(sensv[it*nclass+i]-mean_sensv);
		if (nRUN > 1)
		  stdev_sensv = sqrtf(stdev_sensv/(nRUN-1));



		float mean_specf = 0.0;
		for (it=0; it < nRUN; it++)
			mean_specf += specf[it*nclass+i];
		mean_specf /= nRUN;

		float stdev_specf = 0.0;
		for (it=0; it < nRUN; it++)
			stdev_specf += (specf[it*nclass+i]-mean_specf)*(specf[it*nclass+i]-mean_specf);
		if (nRUN > 1)
		  stdev_specf = sqrtf(stdev_specf/(nRUN-1));

		mean_sensv_all += mean_sensv;
		mean_specf_all += mean_specf;
		fprintf(stdout,"Sensitivity class %d: mean=%f, stdev=%f\n",i+1,mean_sensv,stdev_sensv);
	    fprintf(stdout,"Specificity class %d: mean=%f, stdev=%f\n",i+1,mean_specf,stdev_specf);

	}
	mean_sensv_all /= nclass;
	mean_specf_all /= nclass;

	free(sensv);
	free(specf);

	fprintf(stdout,"Mean sensitivity : %f \n",mean_sensv_all);
	fprintf(stdout,"Mean specificity : %f \n",mean_specf_all);

	fprintf(stdout,"Accuracy of classification is mean=%f, stdev=%f\n",mean,stdev);

	tfinish = iftToc();
	printf("total classification by SVM phase: %f seg\n",iftCompTime(tstart,tfinish)/1000);
}

void iftApplyConvNetworkToImages(char *input_dir, iftConvNetwork *convnet, char *output_dir, int nRUN, float percTrain, float cSVM)
{
  int             number_of_images;
  iftImageNames  *image_names;
  char filename_data[200];

  DIR *dir=NULL;
  iftDataSet *Z;
  omp_set_num_threads(8);
  dir = opendir(output_dir);
  if (dir == NULL){
    char command[200];
    sprintf(command,"mkdir %s",output_dir);
    if (system(command)!=0)
      iftError("Could not open directory","iftApplyConvNetworkToImages");
  }else{
    closedir(dir);
  }
  int xsize[convnet->nstages], ysize[convnet->nstages], zsize[convnet->nstages], nbands[convnet->nstages];
  iftImageDimensionsAlongNetwork(convnet, xsize, ysize, zsize, nbands);
  int nfeatures = xsize[convnet->nstages - 1] * ysize[convnet->nstages - 1] *
    						zsize[convnet->nstages - 1] * nbands[convnet->nstages - 1];

  number_of_images  = iftCountImageNames(input_dir, "mig");
  image_names       = iftCreateAndLoadImageNames(number_of_images, input_dir, "mig");

  Z = iftCreateDataSet(number_of_images, nfeatures);
  
  #pragma omp parallel for shared(Z,number_of_images,image_names,input_dir,convnet)
  for (int s = 0; s < number_of_images ; s++){
    char filename[200];
    sprintf(filename,"%s/%s",input_dir,image_names[s].image_name);
    fprintf(stdout,"Processing %s\n",filename);
    iftMImage *input  = iftReadMImage(filename);
    iftMImage *output = iftApplyConvNetwork(input,convnet);
    Z->sample[s].truelabel = image_names[s].attribute;
    for(int b = 0; b < output->m; b++){
      for(int p = 0; p < output->n; p++){
        Z->sample[s].feat[p + b*output->n] = output->val[p][b];
      }
    }
    iftDestroyMImage(&input);
    iftDestroyMImage(&output);
  }
  Z->nclasses = iftCountNumberOfClasses(image_names, number_of_images);
  sprintf(filename_data,"%s/%s",output_dir,"dataset.data");
  iftWriteOPFDataSet(Z, filename_data);
  iftEvaluateCNN(Z, nRUN, percTrain, cSVM);

  iftDestroyDataSet(&Z);
  iftDestroyImageNames(image_names);
}

int main(int argc, char **argv) 
{
  char            ext[10],*pos;
  timer          *t1=NULL,*t2=NULL;
  iftConvNetwork *convnet;


  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc!=7)
    iftError("Usage: iftCNN <input_directory> <input_parameters.convnet> <output_dir> <NRUN> <PERC_TRAIN> <cSVM(0=OPF)>","main");

  iftRandomSeed(IFT_RANDOM_SEED);

  pos = strrchr(argv[2],'.') + 1;
  sscanf(pos,"%s",ext);

  if (strcmp(ext,"convnet")==0){
    convnet = iftReadConvNetwork(argv[2]);
  }else{
    printf("Invalid file format: %s\n",ext);
    exit(-1);
  }

  t1 = iftTic();
  iftApplyConvNetworkToImages(argv[1],convnet,argv[3],atoi(argv[4]),atof(argv[5]),atof(argv[6]));
  t2 = iftToc();
  fprintf(stdout,"Convnet applied in %f ms\n",iftCompTime(t1,t2));

  iftDestroyConvNetwork(&convnet);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}
