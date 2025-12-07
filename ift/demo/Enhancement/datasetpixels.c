#include "common.h"
#include "ift.h"
#include "iftTrainPixels.h"

int  iftAddLabeledPixelsToDataSetFromMImage(iftMImage* mimg, iftLabelPixelsData* pixelData,iftDataSet* Zpixels,int offsetSample,int offsetID,iftAdjRel* A)
{
	fprintf(stderr,"pixels: %d, pixelsize: %d, dataset size: %d\n",mimg->n,mimg->m,Zpixels->nsamples);

	// collecting pxiels from trainpixel to Z
	int s,sA,b,p,class;
	for(s = 0,sA = 0; s < pixelData->n; s++)
	{
		p     = pixelData->labelPixel[s].p;
		class = pixelData->labelPixel[s].truelabel;
			
		Zpixels->sample[offsetSample+s].truelabel = class;
		Zpixels->sample[offsetSample+s].id    = offsetID + p;
		
		for(b = 0; b < mimg->m ; b++)
			Zpixels->sample[offsetSample+s].feat[b] = mimg->val[p][b];
		sA++;
	}

	return sA;
}


char * replace(
    char   *   original,
    char   *   pattern,
    char   *   replacement
) {
  size_t   replen = strlen(replacement);
  size_t   patlen = strlen(pattern);
  size_t   orilen = strlen(original);

  size_t patcnt = 0;
  const char * oriptr;
  const char * patloc;

  // find how many times the pattern occurs in the original string
  for (oriptr = original; (patloc = strstr(oriptr, pattern)); oriptr = patloc + patlen)
  {
    patcnt++;
  }

  {
    // allocate memory for the new string
    size_t   retlen = orilen + patcnt * (replen - patlen);
    char *   returned = (char *) malloc( sizeof(char) * (retlen + 1) );

    if (returned != NULL)
    {
      // copy the original string, 
      // replacing all the instances of the pattern
      char * retptr = returned;
      for (oriptr = original; (patloc = strstr(oriptr, pattern)); oriptr = patloc + patlen)
      {
        size_t   skplen = patloc - oriptr;
        // copy the section until the occurence of the pattern
        strncpy(retptr, oriptr, skplen);
        retptr += skplen;
        // copy the replacement 
        strncpy(retptr, replacement, replen);
        retptr += replen;
      }
      // copy the rest of the string.
      strcpy(retptr, oriptr);
    }
    return returned;
  }
}



int main(int argc, char **argv) 
{
	if (argc < 4)	iftError("Please provide the following parameters:\n<MSCONVNET_CONFIG_FILE> <TRAIN_PIXELS_FILE> <N_THREADS> [saveMSCONFNET==1]\n\n", "main");

	int idxImg,scale;
	char *file_msconvnet  ;
	char *file_trainpixels;
	char *file_datasetpixels;

	file_msconvnet   = argv[1];
	file_trainpixels = argv[2];
	omp_set_num_threads(atoi(argv[3]));

	printf("Multi-scale convolution network configuration file: %s\n"   , file_msconvnet);
	printf("Train pixels file: %s\n", file_trainpixels);
	puts("");

	iftMSConvNetwork* msconvnet = iftReadMSConvNetwork(file_msconvnet);
	
	int sizeFeatures = 0;
	for(scale = 0; scale < msconvnet->nscales; scale++)
		sizeFeatures += ((msconvnet->convnet[scale])->k_bank[msconvnet->convnet[scale]->nlayers-1])->nkernels;
	printf("feature size: %d\n",sizeFeatures);

	printf("Reading TrainPixels file and corresponding input images\n");
	iftTrainPixels* trainpixels = iftReadTrainPixels(file_trainpixels,1);
	printf("Number of images read: %d\n", trainpixels->nimages);
	
	int totalPixels = 0.;
	for(idxImg = 0; idxImg < trainpixels->nimages ; idxImg++)
		totalPixels += trainpixels->data[idxImg].n;
	printf("%d pixels to be processed\n",totalPixels);
  
	iftAdjRel* A;
	A = iftCircular(trainpixels->radius);

	iftDataSet *Zpixels;
	
	Zpixels = iftCreateDataSet(totalPixels, sizeFeatures);
	Zpixels->nclasses = 2;

	timer *tstart=NULL,*tfinish=NULL;

	printf("\nfeature learning process started with msconvnet parameters\n");
	//preprocessing for paralelism
	int* offsetSample = iftAllocIntArray(trainpixels->nimages);
	int* offsetID     = iftAllocIntArray(trainpixels->nimages);
	for(int idxImg = 0; idxImg < trainpixels->nimages ; idxImg++)
	{
		offsetSample[idxImg] = (idxImg == 0 ? 0 : offsetSample[idxImg-1] + trainpixels->data[idxImg].n);
		offsetID    [idxImg] = (idxImg == 0 ? 0 : offsetID    [idxImg-1] + trainpixels->data[idxImg].image->n);
	}

	tstart = iftTic();
#pragma omp parallel for shared(Zpixels)
	for(int idxImg = 0; idxImg < trainpixels->nimages ; idxImg++)
	{
		timer *t1 = iftTic();

		//        sprintf("%s/%s",dirname,trainpixels->data[idxImg].filename);
		puts("");        
		printf("Processing: (%d/%d) %s (%dx%d)\n",
		  idxImg+1,trainpixels->nimages,trainpixels->data[idxImg].filename,trainpixels->data[idxImg].image->xsize,trainpixels->data[idxImg].image->ysize);

		// Generating Deep Features and Pixels
		timer *tic = iftTic();
		iftMImage* mimg  = iftApplyMSConvNetwork(trainpixels->data[idxImg].image, msconvnet);
		timer *toc = iftToc();
		printf("Feature learning: %f\n", iftCompTime(tic,toc));


		// Creating training SET
		tic = iftTic();
		int npixels   = iftAddLabeledPixelsToDataSetFromMImage(mimg,&(trainpixels->data[idxImg]),Zpixels,offsetSample[idxImg],offsetID[idxImg],A);
		toc = iftToc();
		printf("Adding %d pixels to dataset: %f\n", npixels, iftCompTime(tic,toc));

		iftDestroyMImage(&mimg);

		timer *t2 = iftToc();
		printf("feature extraction: %f seg\n",iftCompTime(t1,t2)/1000);
	}
	iftDestroyAdjRel(&A);
	free(offsetSample);
	free(offsetID);

	if (msconvnet->convnet[0]->with_weights == 3) { /* to be inserted in a new program */
	  fprintf(stderr,"unsuplearning...\n");
	  iftAdjRel* Atmp = msconvnet->convnet[0]->k_bank[0]->A;
	  iftDataSet *Ztmp;
	  iftMMKernel *K;

	  // Learning Only Kernel from Border
	  Ztmp = iftExtractClass(Zpixels,2); // Border class
	  iftSetStatus(Ztmp, IFT_TRAIN);
 	  //K = iftUnsupLearnKernelsFromDataSet(Ztmp,Atmp,0.1,0);
	  K = iftUnsupLearnKernelsByKmedoidsFromDataSet(Ztmp,Atmp,32,1);

	  // Ztmp2 = iftExtractClass(Zpixels,1); // Non-Border class
	  //iftSetStatus(Ztmp2, IFT_TRAIN);
	  //K2 = iftUnsupLearnKernelsFromDataSet(Ztmp2,Atmp,0.1,0);

	  //iftMMKernel* K = iftUnionMMKernel(K1,K2);
	  //iftDestroyMMKernel(&K1);
	  //iftDestroyMMKernel(&K2);

	  iftDestroyDataSet(&Ztmp);
	  //iftDestroyDataSet(&Ztmp2);

	  iftDestroyMMKernel(&msconvnet->convnet[0]->k_bank[0]);
	  msconvnet->convnet[0]->k_bank[0] = K;
	  msconvnet->convnet[0]->nkernels[0] = K->nkernels;
	}

	if (msconvnet->convnet[0]->with_weights == 4) { /* to be inserted in a new program */
	  fprintf(stderr,"whitening dataset...\n");
	  iftSetStatus(Zpixels, IFT_TRAIN);
	  iftDataSet* Zaux = iftWhiteningTransform(Zpixels);
	  iftDestroyDataSet(&Zpixels);
	  Zpixels = Zaux;

	  if (Zpixels->fsp.W != NULL){
	    iftAdjRel* Atmp = msconvnet->convnet[0]->k_bank[0]->A;
	    iftMMKernel* K = iftRandomMMKernel(Atmp,msconvnet->convnet[0]->input_nbands,msconvnet->convnet[0]->nkernels[0]);

	    K->W     = iftCopyMatrix(Zpixels->fsp.W);
	    K->mean  = iftAllocFloatArray(Zpixels->nfeats);
	    K->stdev = iftAllocFloatArray(Zpixels->nfeats);
	    for (int i=0; i < Zpixels->nfeats; i++) {
	      K->mean[i]  = Zpixels->fsp.mean[i];
	      K->stdev[i] = Zpixels->fsp.stdev[i];
	    }

	    iftDestroyMMKernel(&msconvnet->convnet[0]->k_bank[0]);
	    msconvnet->convnet[0]->k_bank[0] = K;
	    msconvnet->convnet[0]->nkernels[0] = K->nkernels;
	  }
	}

	if ( (argc > 4) && (atoi(argv[4]) == 1) )
	{
	  // testing iftWriteMSConvNetwork function
	  for(scale = 0; scale < msconvnet->nscales; scale++)
	    msconvnet->convnet[scale]->with_weights = 1;
	  iftWriteMSConvNetwork(msconvnet,"msconvnet.out");
	  iftDestroyMSConvNetwork(&msconvnet);
	  msconvnet = iftReadMSConvNetwork("msconvnet.out");
	}

	fprintf(stderr,"destroying MSConvNetwork...\n");
	iftDestroyMSConvNetwork(&msconvnet);

	fprintf(stderr,"destroying TrainPixels...\n");
	iftDestroyTrainPixels(&trainpixels);

	tfinish = iftToc();
	printf("total feature extraction: %f seg\n",iftCompTime(tstart,tfinish)/1000);

	tstart = iftTic();	
	file_datasetpixels = replace(file_trainpixels,"TrainPixels","DataSetPixels");
	iftWriteOPFDataSet(Zpixels,file_datasetpixels);
	tfinish = iftToc();
	printf("saving DataSetPixel: %f seg\n",iftCompTime(tstart,tfinish)/1000);
	
	fprintf(stderr,"destroying DataSet...\n");
	iftDestroyDataSet(&Zpixels);

	fprintf(stderr,"destroyed...\n");
	free(file_datasetpixels);

	return 0;
}
