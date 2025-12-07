#include "common.h"
#include "ift.h"
#include "iftTrainPixels.h"

#define BORDER_CLASSIFY_DEBUG

int main(int argc, char **argv) 
{
	if (argc != 8)	iftError("Please provide the following parameters:\n<MSCONVNET_CONFI_FILE> <TRAIN_PIXELS_FILE> <DATASET_TRAIN_PIXELS_FILE> <FILE_TEST_IMAGE_NAMES> <FILE_TEST_GT_IMAGE_NAMES> <cSVM(0=OPF)> <N_THREADS>\n\n", "main");


	int idxImg,totalPixels;
	int nTrainSamples,nTestImages,nTestGTImages;
	float cSVM,percTrain;
	char *file_msconvnet  ;
	char *file_trainpixels;
	char *file_datasettrainpixels;
	char *file_testimageNames;
	char *file_testGTimageNames;
	char *folder_testname;
	char *folder_testGTname;
	FileNames * filestest = NULL;
	FileNames * filestestGT = NULL;

	file_msconvnet          = argv[1];
	file_trainpixels        = argv[2];
	file_datasettrainpixels = argv[3];
	file_testimageNames     = argv[4];
	file_testGTimageNames   = argv[5];
	cSVM      = atof(argv[6]);
	omp_set_num_threads(atoi(argv[7]));

	percTrain = 1.0; /* all data is used for training */
	nTrainSamples = 8000;


	printf("MSConvNet config file: %s\n",file_msconvnet);
	printf("Train pixels file: %s\n", file_trainpixels);
	printf("DataSet pixels file: %s\n", file_datasettrainpixels);
	puts("");

	// reading msconvnet parameters
	printf("Reading MSConvNet file\n");
	iftMSConvNetwork* msconvnet = iftReadMSConvNetwork(file_msconvnet);
	
	int sizeFeatures = 0;
	for(int scale = 0; scale < msconvnet->nscales; scale++)
		sizeFeatures += ((msconvnet->convnet[scale])->k_bank[msconvnet->convnet[scale]->nlayers-1])->nkernels;
	printf("feature size: %d\n",sizeFeatures);
	printf("\n");

	// Reading TrainPixels file
	printf("Reading TrainPixels file\n");
	iftTrainPixels* trainpixels = iftReadTrainPixels(file_trainpixels,0);

	for(totalPixels = 0,idxImg = 0; idxImg < trainpixels->nimages ; idxImg++)
    	totalPixels += trainpixels->data[idxImg].n;
	printf("%d pixels to be trained\n",totalPixels);

	// Reading file names of Testing and GT images
	printf("Reading file names of testing and GT images\n");
	folder_testname   = getFolderName(file_testimageNames);
	folder_testGTname = getFolderName(file_testGTimageNames);
	nTestImages   = countImages(file_testimageNames);
	nTestGTImages = countImages(file_testGTimageNames);
	if (nTestImages != nTestGTImages){
	  char msg[100];
	  sprintf(msg,"Number of images in FILE_TEST_IMAGE_NAMES (%d) and FILE_TEST_GT_IMAGE_NAMES (%d)\n\n\n",nTestImages,nTestGTImages);
	  iftError(msg, "main");
	}
	filestest   = createFileNames(nTestImages);
	filestestGT = createFileNames(nTestImages);
	loadFileNames(filestest  , file_testimageNames );
	loadFileNames(filestestGT, file_testGTimageNames);

	timer *tstart, *tfinish;
	timer *tic,*toc;

	// Effectively loading/reading training dataset file
	printf("Reading Training DataSet file\n");
	tic = iftTic();
	iftDataSet *Zpixels = iftReadOPFDataSet(file_datasettrainpixels);
	toc = iftToc();
	fprintf(stderr,"training dataset reading: %f\n", iftCompTime(tic,toc));
	
	if (totalPixels != Zpixels->nsamples)	{
	  char msg[100];
	  sprintf(msg,"Number of pixels from TrainPixels (%d) and DataSetPixels (%d) are different\n\n",totalPixels,Zpixels->nsamples);
	  iftError(msg, "main");
	}
  
	puts("");puts("");	
	printf("\ntraining/learning classification phase (cSVM:%.2f,pTrain:%6.2f)\n",cSVM,100*percTrain);
	tstart = iftTic();

	iftSVM      *svm   = NULL;
	iftCplGraph *graph = NULL;
	iftDataSet *Ztrain[2];

	tic = iftTic();
	// Selecting training images - leave pixels out of %(1-percTrain) images
	// 60% of border pixels for OPF training, empirically chosen
	iftSelectSupTrainPixels(Zpixels,trainpixels,percTrain,0.5); // 10% of border pixels
	Ztrain[0] = iftExtractSamples(Zpixels,IFT_TRAIN);
	Ztrain[1] = iftNormalizeDataSet(Ztrain[0]);
	iftDestroyDataSet(&Ztrain[0]);

	toc = iftToc();
	fprintf(stderr,"Training dataset preprocessing (%d samples): %f\n",Ztrain[1]->nsamples,iftCompTime(tic,toc));

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
	  iftSelectSupTrainSamples(Ztrain[1],(float)nTrainSamples/Ztrain[1]->nsamples);
	  graph = iftSupLearn(Ztrain[1]);
	  // Using [0] as tmp for creating a "training" dataset
	  Ztrain[0] = iftExtractSamples(Ztrain[1],IFT_TRAIN);
	  iftDestroyDataSet(&Ztrain[1]);
	  Ztrain[1] = Ztrain[0]; Ztrain[0] = NULL;
	  iftDestroyCplGraph(&graph);
	  graph = iftCreateCplGraph(Ztrain[1]);          // Training
	  iftSupTrain(graph);
	}
	toc = iftToc();
	fprintf(stderr,"training classifier (%d samples): %f\n", Ztrain[1]->nsamples,iftCompTime(tic,toc));


#ifdef BORDER_CLASSIFY_DEBUG
	mkdir("output/",S_IRWXU);
#endif

	float* fscore  = iftAllocFloatArray(nTestImages); 
#pragma omp parallel for shared(fscore,svm,graph,cSVM)
	for(int idxImg = 0; idxImg < nTestImages ; idxImg++)
	{
		timer *t1 = iftTic();

		char fullPath[256];fullPath[0] = '\0';
		strcat(fullPath,folder_testname);
		strcat(fullPath,filestest[idxImg].filename);
		//printf("%s\n",fullPath);

		iftImage *imgin;
		if (strstr(fullPath,".ppm") != NULL)
		  imgin = iftReadImageP6(fullPath);
		else
		  imgin = iftReadImageP5(fullPath);
		puts("");        
		printf("Processing: (%d/%d) %s (%dx%d)\n",
		  idxImg+1,nTestImages,filestest[idxImg].filename,imgin->xsize,imgin->ysize);

		//iftMImage* mimgin = iftImageToMImage(img,RGB_CSPACE);
		iftMImage* mimgin = iftImageToMImage(imgin,YCbCr_CSPACE);

		fullPath[0] = '\0';
		strcat(fullPath,folder_testGTname);
		strcat(fullPath,filestestGT[idxImg].filename);
		iftImage *imgGT = iftReadImageP5(fullPath);

		// Generating Deep Features and Pixels
		timer *tic2 = iftTic();
		iftMImage* mimgout  = iftApplyMSConvNetwork(mimgin, msconvnet);
		iftDestroyMImage(&mimgin);
		timer *toc2 = iftToc();
		printf("Feature learning: %f\n", iftCompTime(tic2,toc2));

		// Creating dataset for classifying
		iftDataSet *Ztest[2];
		tic2 = iftTic();
		Ztest[0] = iftMImageToDataSet(mimgout);
		iftImageGTToDataSet(imgGT,Ztest[0]);
		iftDestroyMImage(&mimgout);
		iftDestroyImage(&imgGT);
		toc2 = iftToc();
		printf("Creating dataset for classifying: %f\n", iftCompTime(tic2,toc2));

		tic2 = iftTic();
		iftSetStatus(Ztest[0],IFT_TEST);
		Ztest [1] = iftNormalizeTestDataSet(Ztrain[1],Ztest[0]);
		iftDestroyDataSet(&Ztest [0]);
		toc2 = iftToc();
		printf("Normalizing dataset for classifying: %f\n", iftCompTime(tic2,toc2));

		tic2 = iftTic();
		if (cSVM > 0.0)
			iftSVMLinearClassifyOVA(svm, Ztest[1], IFT_TEST, NULL); // Classification
		else if (cSVM < 0.0)
			iftSVMClassifyOVA(svm, Ztest[1], IFT_TEST); // Classification
		else
			iftBorderClassify(graph,Ztest[1],2);  // Classify test samples - 1 means non-border at numerator for computing the edges
		toc2 = iftToc();
		fprintf(stderr,"Classifying the test set: %f\n", iftCompTime(tic2,toc2));

		timer *t2 = iftToc();
		printf("classification time: %f seg\n",iftCompTime(t1,t2)/1000);

		// iftSkewedTruePositives(Ztest[1]);
		fscore[idxImg] = iftFscore(Ztest[1],2); // Compute fscore on test set
		printf("fscore[%d] = %f\n",idxImg,fscore[idxImg]);

#ifdef BORDER_CLASSIFY_DEBUG
		iftAdjRel *A = iftCircular(sqrtf(2.0));
		//iftImage* imgout = iftPixelDataSetToImage(Ztest[1],imgin);
		iftFImage* imgoutF = iftPixelDataSetToFImage(Ztest[1],imgin);
		iftImage* imgout = iftFImageToImage(imgoutF,255);

		char imgpath[256];
		sprintf(imgpath,"output/%s",filestestGT[idxImg].filename);
		iftWriteImageP5(imgout,imgpath);

		sprintf(imgpath,"output/%s",filestestGT[idxImg].filename);
		char* pch = strrchr(imgpath,'.');
		if (pch== NULL)
		  iftError("error in image path output","borderclassify");
		strcpy(pch,".grad.npy");
		iftWriteFImage(imgoutF,imgpath);

		iftImage* marker = iftVolumeClose(imgout,100);
		iftImage* label  = iftWaterGray(imgout,marker,A);
		iftWaterGrayFBorder(imgoutF,label,A);

		sprintf(imgpath,"output/%s",filestestGT[idxImg].filename);
		pch = strrchr(imgpath,'.');
		if (pch== NULL)
		  iftError("error in image path output","borderclassify");
		strcpy(pch,".thin.npy");
		iftWriteFImage(imgoutF,imgpath);


		iftDestroyFImage(&imgoutF);
		iftDestroyImage(&imgout);
		iftDestroyImage(&marker);
		iftDestroyImage(&label);
		iftDestroyAdjRel(&A);
#endif	      
		iftDestroyImage(&imgin);

		iftDestroyDataSet(&Ztest[1]);    
	}

	float mean = 0.0;
	for (idxImg=0; idxImg < nTestImages; idxImg++)
		mean += fscore[idxImg];
	mean /= nTestImages;

	float stdev = 0.0;
	for (idxImg=0; idxImg < nTestImages; idxImg++)
	  stdev += (fscore[idxImg]-mean)*(fscore[idxImg]-mean);
	if (nTestImages > 1)
	  stdev = sqrtf(stdev/(nTestImages-1));

	free(fscore); 

	fprintf(stdout,"Fscore of classification is mean=%f, stdev=%f\n",mean,stdev); 

	tfinish = iftToc();
	printf("total classification by SVM phase: %f seg\n",iftCompTime(tstart,tfinish)/1000);

	if (cSVM != 0.0)
	  iftDestroySVM(svm);
	if (graph != NULL)
	  iftDestroyCplGraph(&graph);

	fprintf(stderr,"destroying MSConvNetwork...\n");
	iftDestroyMSConvNetwork(&msconvnet);

	fprintf(stderr,"destroying Training Dataset...\n");
	iftDestroyDataSet(&Zpixels);
	iftDestroyDataSet(&(Ztrain[1]));

	fprintf(stderr,"destroying TrainPixels...\n");
	iftDestroyTrainPixels(&trainpixels);

	free(folder_testname);
	free(folder_testGTname);
	destroyFileNames(filestest);
	destroyFileNames(filestestGT);

	return 0;
}
