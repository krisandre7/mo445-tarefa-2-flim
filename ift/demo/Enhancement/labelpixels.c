#include "common.h"
#include "ift.h"
#include "iftTrainPixels.h"

iftDataSet* iftMImageToPixelDataSet(iftMImage* mimg) 
{ 

	iftDataSet* Zpx = iftCreateDataSet(mimg->n, mimg->m);
	Zpx->nclasses = 2;
	
	int p,b;
	for(p = 0; p < mimg->n; p++)
	{
		for(b = 0; b < mimg->m ; b++)
			Zpx->sample[p].feat[b] = mimg->val[p][b];
		Zpx->sample[p].id = p;
		Zpx->sample[p].status = IFT_TEST;
	}
	return Zpx;
}

iftFImage* iftPixelDataSetToFImage(iftDataSet* dataset,iftMImage* mimg)
{
	printf("ini iftPixelDataSetToFImage \n");
	printf("img pixels %d\n", mimg->n);
	printf("nsamples %d\n", dataset->nsamples);
	fprintf(stderr,"(%d x %d):\n",mimg->n,dataset->nsamples);
	if (mimg->n != dataset->nsamples)
		iftError("incompatible dimensions ", "iftEdgesDataSetToImage");
	
	iftFImage* output = iftCreateFImage(mimg->xsize,mimg->ysize,mimg->zsize);
	
	int s;
	for(s = 0;s < dataset->nsamples; s++)
	{
		if (dataset->sample[s].label == 2) // edges
		{
			if (dataset->sample[s].weight > 0)
			{
				output->val[s] = dataset->sample[s].weight;
			}
		}
	}
	printf("after for \n");
	return output;
}


int iftSelectSupTrainPixels(iftDataSet *Z, iftTrainPixels* trainpixels,float train_perc)
{ 
  int s, *imageS=NULL,*imageN=NULL,*count=NULL, i;
  int t, high, nsamples, c, *class_size=NULL;

  if (Z->nclasses == 0)
    iftError("There are no classes","iftSelectSupTrainPixels");

  if ((train_perc <= 0.0)||(train_perc > 1.0))
    iftError("Invalid percentage of training samples","iftSelectSupTrainPixels");

  // Reset status
  iftSetStatus(Z, IFT_TEST);

  // Verify if it is the trivial case of selecting all samples.
  if (train_perc == 1.0) {
    for (s=0; s < Z->nsamples; s++) 
      Z->sample[s].status = IFT_TRAIN;
    Z->ntrainsamples = Z->nsamples;  
    return(Z->nsamples);
  }

  // Compute the number of training samples
  Z->ntrainsamples = (int) (train_perc*Z->nsamples);
  if (Z->ntrainsamples == 0)
    iftError("Percentage is too low for this dataset","iftSelectSupTrainPixels");
  
  int ntrainimages = (int)(train_perc * trainpixels->nimages + 0.5);
  if (ntrainimages == 0)
    iftError("Percentage is too low for this trainpixels set","iftSelectSupTrainPixels");

  // Count number of samples per class
  class_size = iftAllocIntArray(Z->nclasses+1);
  for (s=0; s < Z->nsamples; s++){
    class_size[Z->sample[s].truelabel]++;
  }

  // Verify percentage and number of training samples per class - in this case the iftTranPixels structure should guarantee that property
  Z->ntrainsamples = 0;
  for (c=1; c <= Z->nclasses; c++) {
    nsamples = (int)(train_perc*class_size[c]);
    if (nsamples > class_size[c] || nsamples == 0)
      nsamples = class_size[c];
    Z->ntrainsamples += nsamples;
    if (nsamples <= 0){
      fprintf(stderr,"For class %d\n",c);
      iftError("No available samples","iftSelectSupTrainPixels");
    }
  }
  
  // Randomly select images 
  ntrainimages = (int)(train_perc * trainpixels->nimages + 0.5);
  imageS = iftAllocIntArray(trainpixels->nimages);
  imageN = iftAllocIntArray(trainpixels->nimages);
  count = iftAllocIntArray(trainpixels->nimages); 
  for (i=0; i < trainpixels->nimages; i++) {
      imageS[i]= (i == 0) ? 0 : imageS[i-1]+trainpixels->data[i-1].n;
      imageN[i]= trainpixels->data[i].n; 
      count[i] =100;
  }
  t = 0; high = trainpixels->nimages-1;
  while (t < ntrainimages) {
    i = iftRandomInteger(0,high);
    if (count[i]==0){
      // then select all samples from that image
      for(s=imageS[i]; s < imageS[i]+imageN[i]; s++)
	Z->sample[s].status=IFT_TRAIN;
      
      iftSwitchValues(&imageS[i],&imageS[high]);
      iftSwitchValues(&imageN[i],&imageN[high]);
      iftSwitchValues(&count [i],&count [high]);
      t++; high--;
    }else{
      count[i]--;
    }
  }
  free(imageS);free(imageN);free(count);
  free(class_size);

  return(Z->ntrainsamples);
}


int main(int argc, char **argv) 
{
	if (argc != 7)	iftError("Please provide the following parameters:\n<TRAIN_PIXELS_FILE> <DATASET_PIXELS_FILE> <MSCONVNET_FILE> <FILE_INPUT_NAMES> <cSVM(0=OPF)> <N_THREADS>\n\n", "main");

	iftSVM      *svm   = NULL;
	iftCplGraph *graph = NULL;

	int idxImg,totalPixels;
	int it;
	float cSVM;
	char *file_trainpixels;
	char *file_datasetpixels;
	char *file_inputImages;
	char *file_msconvnet  ;

	file_trainpixels   = argv[1];
	file_datasetpixels = argv[2];
	file_msconvnet = argv[3];
	file_inputImages   = argv[4];
	cSVM      = atof(argv[5]);
	omp_set_num_threads(atoi(argv[6]));

	printf("Train pixels file: %s\n", file_trainpixels);
	printf("DataSet pixels file: %s\n", file_datasetpixels);	
	puts("");

	printf("Reading TrainPixels file and corresponding input images\n");
	iftTrainPixels* trainpixels = iftReadTrainPixels(file_trainpixels);
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
	
	timer *tic, *toc, *tstart, *tfinish;
	puts("");puts("");	
	printf("\ntraining and learning classification phase (cSVM:%.2f)\n",cSVM);
	tstart = iftTic();
	

	iftDataSet *Ztrain[2], *Ztest[2];
	//float* acc  = iftAllocFloatArray(nRUN); 

	// Read MSCONVNET
	iftMSConvNetwork* msconvnet = iftReadMSConvNetwork(file_msconvnet);

	// Read Input images (Images to label) 
	int nimages = 0;
	char *folder_name = iftAllocCharArray(256);
	char fullPath[128];
	FileNames * files = NULL;
	
	folder_name = getFolderName(file_inputImages);
	nimages = countImages(file_inputImages);
	files = createFileNames(nimages);
	loadFileNames(files, file_inputImages);

	printf("Input Folder: %s\n", folder_name);
	printf("Number of Images: %d\n", nimages);
	printf("File with image names: %s\n"   , file_inputImages);
	puts("");


	// For all input images
	for( idxImg = 0; idxImg < nimages; idxImg++)
	{
		// Read input images
		fullPath[0] = 0;
		strcpy(fullPath, folder_name);
		strcat(fullPath, files[idxImg].filename);
		iftImage *img = iftReadImageP5(fullPath);

		// Extract Deep Features
		iftMImage *mimg = iftImageToMImage(img,RGB_CSPACE);
		iftMImage *msimg = iftApplyMSConvNetwork(mimg, msconvnet);
		iftDataSet* Zimg = iftMImageToPixelDataSet(msimg);

		// Select Train Samples
		tic = iftTic();
		iftSelectSupTrainSamples(Zpixels,(float)0.999); // Selecting training samples
		iftSetStatus(Zpixels, IFT_TRAIN);
		//iftSelectSupTrainPixels(Zpixels,trainpixels,percTrain); // Selecting training images - leave pixels out of %(1-percTrain) images
		
		Ztrain[0] = iftExtractSamples(Zpixels,IFT_TRAIN);
		Ztest [0] = iftExtractSamples(Zimg,IFT_TEST);

		Ztrain[1] = iftNormalizeDataSet(Ztrain[0]);
		iftDestroyDataSet(&Ztrain[0]);
		Ztest [1] = iftNormalizeTestDataSet(Ztrain[1],Ztest[0]);
		iftDestroyDataSet(&Ztest [0]);
		toc = iftToc();
		fprintf(stderr,"selecting training(%d)/testing(%d) data: %f\n",Ztrain[1]->nsamples,Ztest[1]->nsamples,iftCompTime(tic,toc));
		
		tic = iftTic();	
		if (cSVM != 0.0) {
			svm=iftCreateLinearSVC(cSVM);
			iftSVMTrainOVA(svm, Ztrain[1]); // Training
		}
		else {
			graph = iftCreateCplGraph(Ztrain[1]);          // Training
			iftSupTrain(graph);
		}
		toc = iftToc();
		fprintf(stderr,"training svm classifier: %f\n", iftCompTime(tic,toc));

	
		tic = iftTic();
		if (cSVM != 0.0)
			iftSVMLinearClassifyOVA(svm, Ztest[1], IFT_TEST); // Classification
		else
			iftClassify(graph,Ztest[1]);                  // Classify test samples
		toc = iftToc();
		fprintf(stderr,"classifying the test set: %f\n", iftCompTime(tic,toc));

		//acc[it] = iftSkewedTruePositives(Ztest[1]); // Compute accuracy on test set
		//printf("acc[%d] = %f\n",it,acc[it]);


		// write labeled image
		iftFImage* imgedges = iftPixelDataSetToFImage(Ztest[1],mimg);
		iftImage* imgout = iftFImageToImage(imgedges, 4095);
		
	  	fullPath[0] = 0;
	  	strcpy(fullPath, folder_name);
	  	strcat(fullPath, files[idxImg].filename);
	  	strcat(fullPath, ".lbl.pgm");

		iftWriteImageP2(imgout,fullPath);



		iftDestroyDataSet(&Ztrain[1]); // due to OPF usage, train data should be destroy only after test
		iftDestroyDataSet(&Ztest[1]);    


		if (cSVM != 0.0)
			iftDestroySVM(svm);
		if (graph != NULL)
			iftDestroyCplGraph(&graph);
	}

	iftDestroyDataSet(&Zpixels);


	return 0;
}
