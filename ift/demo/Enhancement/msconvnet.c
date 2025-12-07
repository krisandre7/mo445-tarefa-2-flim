#include "common.h"
#include "ift.h"

//And here it begins....
int main(int argc, char **argv) 
{
	if (argc != 5)	iftError("Please provide the following parameters:\n<CONFIG_FILE> <FILE_IMAGE_NAMES> <FILE_IMAGE_NAMES_GT> <N_THREADS>\n\n", "main");
	
	int idxImg,s;
	int nimages = 0,nimagesGT = 0;
	char fullPath[128];
	char *file_config       ; // = iftAllocCharArray(64);
	char *file_imageNames   ; // = iftAllocCharArray(64);
	char *file_imageNamesGT ; // = iftAllocCharArray(64);	
	char *folder_name       = iftAllocCharArray(256);
	char *folder_nameGT     = iftAllocCharArray(256);
	FileNames * files = NULL;
	FileNames * filesGT = NULL;	

	//FIXME
	omp_set_num_threads(atoi(argv[4]));

	file_config       = argv[1];
	file_imageNames   = argv[2];
	file_imageNamesGT = argv[3];

	folder_name = getFolderName(file_imageNames);
	nimages = countImages(file_imageNames);
	
	folder_nameGT = getFolderName(file_imageNamesGT);
	nimagesGT = countImages(file_imageNamesGT);
	
	if ( nimages != nimagesGT )
		iftError("Number of files specified in original and gt sets are different\n\n", "main");

	files = createFileNames(nimages);
	loadFileNames(files, file_imageNames);
	
	filesGT = createFileNames(nimagesGT);
	loadFileNames(filesGT, file_imageNamesGT);	

	printf("Folder: %s\n", folder_name);
	printf("Number of Images: %d\n", nimages);
	printf("File with image names: %s\n"   , file_imageNames);
	printf("File with GT image names: %s\n", file_imageNamesGT);
	puts("");puts("");
	

	//FIXME
	//iftMSConvNetwork *msconvnet = parseConfigFile(file_config);
	iftMSConvNetwork *msconvnet = iftReadMSConvNetwork(file_config);
	

//	#pragma omp parallel for private(fullPath)
	for(idxImg = 0; idxImg < nimages ; idxImg++)
	{
		timer* t1, *t2;
		iftDataSet *Z1,*Z2,*Zl;
		t1 = iftTic();
		
		fullPath[0] = 0;
		strcpy(fullPath, folder_name);
		strcat(fullPath, files[idxImg].filename);
		printf("Processing: (%d/%d) %s/%s\n", idxImg+1,nimages,files[idxImg].filename,filesGT[idxImg].filename);

		fprintf(stderr,"%s\n",fullPath);
		timer *tic, *toc;
		tic = iftTic();	
		iftImage *img = iftReadImageP5(fullPath);
		
		// Creating imgWG
		float spatial_radius = 1.;
		float volume_threshold = 1000.;
		iftAdjRel* adj_relation = iftCircular(spatial_radius);
		iftImage* basins = iftImageBasins(img,adj_relation);
		
		int nregions = 257;
		iftImage* marker=NULL; iftImage* imgWG=NULL;
		while(nregions > 256)
		{
			if (marker != NULL) iftDestroyImage(&marker);
			marker = iftVolumeClose(basins,volume_threshold);
			if (imgWG  != NULL) iftDestroyImage(&imgWG);
			imgWG = iftWaterGray(basins,marker,adj_relation);
			nregions = iftMaximumValue(imgWG);
			fprintf(stderr,"volume_threshold: %f, nregions: %d\n",volume_threshold,nregions);
			volume_threshold *= 2;
		}
		volume_threshold /= 2;
		iftWriteImageP2(imgWG,"watergray.pgm");

		iftDestroyImage(&basins);
		iftDestroyImage(&marker);
		iftDestroyAdjRel(&adj_relation);


		// Reading imgGT
		fullPath[0] = 0;
		strcpy(fullPath, folder_nameGT);
		strcat(fullPath, filesGT[idxImg].filename);
		iftImage *imgGT = iftReadImageP5(fullPath);
		
		fprintf(stderr,"(%d,%d) : (%d,%d)\n",img->xsize,img->ysize,imgGT->xsize,imgGT->ysize);
		iftVerifyImageDomains(img,imgGT,"main");
		fprintf(stderr,"(%d,%d) : (%d,%d)\n",img->xsize,img->ysize,imgWG->xsize,imgWG->ysize);
		iftVerifyImageDomains(img,imgWG,"main");


		// Generating Deep Features and Edges
		iftMImage* imimg = iftImageToMImage(img,YCbCr_CSPACE);
		iftMImage*  mimg = iftApplyMSConvNetwork(imimg, msconvnet);

		float radius = 1.0;
		iftAdjRel* A;
		if (mimg->zsize == 1)
			A = iftCircularEdges(radius);
		else
			A = iftSphericEdges(radius);
		
		// Creating training SET
		tic = iftTic();
		iftDataSet* Zedges = iftMImageToLabeledEdgesDataSet(mimg,imgGT,imgWG,A);
		toc = iftToc();
		printf("Dataset creation: %f\n", iftCompTime(tic,toc));



		tic = iftTic();

		Z1 = iftExtractClass(Zedges,1);
		printf("class 1: %d\n",Z1->nsamples);
		
		Z2 = iftExtractClass(Zedges,2);
		printf("class 2: %d\n",Z2->nsamples);

		float fData = 2.; // (float)Z2->nsamples/Z1->nsamples; // 20.;	

		int selected = iftSelectSupTrainSamples(Z1,(float)Z2->nsamples/Z1->nsamples*fData);
		fprintf(stderr,"class (2) selected: %d\n",selected);

		iftDestroyDataSet(&Zedges);
		Zedges = iftCreateDataSet(Z2->nsamples+selected,Z2->nfeats);
		Zedges->nclasses = 2;
		for( s = 0; s < Z2->nsamples; s++)
		{
			Zedges->sample[s].truelabel = Z2->sample[s].truelabel;
			memcpy(Zedges->sample[s].feat,(Z2->sample[s].feat),sizeof(float)*Zedges->nfeats);
		}
		int j;
		for( s = 0,j = 0; s < Z1->nsamples; s++)
		{
			if (Z1->sample[s].status == IFT_TRAIN)
			{	Zedges->sample[Z2->nsamples+j].truelabel = Z1->sample[s].truelabel;
				memcpy(Zedges->sample[Z2->nsamples+j].feat,(Z1->sample[s].feat),sizeof(float)*Zedges->nfeats); j++;}
		}
		iftDestroyDataSet(&Z1);
		iftDestroyDataSet(&Z2);

		selected = iftSelectSupTrainSamples(Zedges,(float)1.0);
		
		printf("for final dataset - selected: %d\n",selected);
		Zl = iftCreateDataSet(selected,Zedges->nfeats);
		Zl->nclasses = 2;
		for( s = 0,j = 0; s < Zedges->nsamples; s++)
		{
			if (Zedges->sample[s].status == IFT_TRAIN)
			{	Zl->sample[j].truelabel = Zedges->sample[s].truelabel;
				memcpy(Zl->sample[j].feat,(Zedges->sample[s].feat),sizeof(float)*Zedges->nfeats); 
				j++;
			}
		}
		iftDestroyDataSet(&Zedges);
		Zedges = iftCopyDataSet(Zl);
		
		toc = iftToc();
		printf("Dataset reduction/selection(%3.2f): %f\n", fData,iftCompTime(tic,toc));

		// loop
		int it;
		timer* tstart, *tfinish;
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
			iftSelectSupTrainSamples(Zedges,0.9); // Selecting training samples

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
			iftSVM *svm=iftCreateLinearSVC(C);
			iftSVMTrainOVA(svm, Ztrain[1]); // Training
			iftDestroyDataSet(&Ztrain[1]);
			toc = iftToc();
			fprintf(stderr,"training svm classifier: %f\n", iftCompTime(tic,toc));

		
			tic = iftTic();
			//iftSetStatus(Ztrain[1], IFT_TEST);
			//iftSVMLinearClassifyOVA(svm, Ztrain[1], IFT_TEST); // Classification
			//iftSVMClassifyOVA(svm, Ztrain[1], IFT_TEST); // Classification
			iftSVMLinearClassifyOVA(svm, Ztest[1], IFT_TEST); // Classification
			//iftSVMClassifyOVA(svm, Ztest[1], IFT_TEST); // Classification
			toc = iftToc();
			fprintf(stderr,"classifying the test set: %f\n", iftCompTime(tic,toc));

			//acc[it] = iftSkewedTruePositives(Ztrain[1]); // Compute accuracy on test set
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
		iftDestroyDataSet(&Zedges);

		puts("");

		// selecting all data for training
		iftSelectSupTrainSamples(Zl,1.0);
		Zedges = iftNormalizeDataSet(Zl);
		iftDestroyDataSet(&Zl);
		fprintf(stderr,"data normalization\n");

		// Training SVM
		float C = 1e5;
		tic = iftTic();
		iftSVM *svm=iftCreateLinearSVC(C);
		iftSVMTrainOVA(svm, Zedges); // Training
		toc = iftToc();
		fprintf(stderr,"training dataset: %f\n", iftCompTime(tic,toc));
		
		
		// Extracting features for testing
		tic = iftTic();
		Ztest[0]  = iftMImageToEdgesDataSet(mimg,A);
		toc = iftToc();
		fprintf(stderr,"extracting features for the entire image: %f\n", iftCompTime(tic,toc));	
		
		tic = iftTic();
		Ztest[1] = iftNormalizeTestDataSet(Zedges,Ztest[0]);
		toc = iftToc();
		fprintf(stderr,"new data normalization: %f\n", iftCompTime(tic,toc));
		iftDestroyDataSet(&Zedges);
		iftDestroyDataSet(&Ztest[0]);

		tic = iftTic();
		iftSVMLinearClassifyOVA(svm, Ztest[1], IFT_TEST); // Classification
		//iftSVMClassifyOVA(svm, Ztest[1], IFT_TEST); // Classification
		toc = iftToc();
		fprintf(stderr,"classifying the entire image: %f\n", iftCompTime(tic,toc));
		
		iftFImage* imgedges = iftEdgesDataSetToFImage(Ztest[1],mimg,A);
		//iftWriteFImage(imgedges,"edgesF.pgm");
		iftImage* imgout = iftFImageToImage(imgedges, 4095);
		iftDestroyDataSet(&Ztest[1]);
		iftDestroyAdjRel(&A);
	
	  // 	fullPath[0] = 0;
	  // 	strcpy(fullPath, folder_name);
	  // 	strcat(fullPath, files[0].filename);

		iftWriteImageP2(imgout,"edges.pgm");
  
		t2 = iftToc();

		puts("");
		printf("Finished. Time: %f seg\n",iftCompTime(t1,t2)/1000);

		iftDestroyImage(&img);
		iftDestroyImage(&imgGT);
		iftDestroyImage(&imgWG);
		iftDestroyMImage(&mimg);
		iftDestroyMImage(&imimg);
	}

// 	// Training SVM
// 	float C = 1e5;
// 	tic = iftTic();
// 	iftSVM *svm=iftCreateLinearSVC(C);
// 	iftSVMTrainOVO(svm, Zedges); // Training
// 	toc = iftToc();
// 	fprintf(stderr,"training dataset: %f\n", iftCompTime(tic,toc));
// 
// 
// 	
// 	// Extracting features for testing
// 	tic = iftTic();
// 	iftDataSet* Ztest  = iftMImageToEdgesDataSet(mimg,A);
// 	toc = iftToc();
// 	fprintf(stderr,"extracting features for the entire image: %f\n", iftCompTime(tic,toc));	
// 	
// 	tic = iftTic();
// 	iftDataSet* ZtestN = iftNormalizeTestDataSet(Zedges,Ztest);
// 	toc = iftToc();
// 	fprintf(stderr,"new data normalization: %f\n", iftCompTime(tic,toc));
// 	iftDestroyDataSet(&Zedges);
// 	iftDestroyDataSet(&Ztest);
// 
// 	tic = iftTic();
// 	iftSVMLinearClassifyOVO(svm, ZtestN, IFT_TEST); // Classification
// 	toc = iftToc();
// 	fprintf(stderr,"classifying the entire image: %f\n", iftCompTime(tic,toc));
// 
// 	
// 	iftFImage* imgedges = iftEdgesDataSetToFImage(ZtestN,mimg,A);
// 	//iftWriteFImage(imgedges,"edgesF.pgm");
// 	iftImage* imgout = iftFImageToImage(imgedges, 4095);
// 	iftDestroyDataSet(&ZtestN);
// 	iftDestroyAdjRel(&A);
// 	
// // 	fullPath[0] = 0;
// // 	strcpy(fullPath, folder_name);
// // 	strcat(fullPath, files[0].filename);
// 
// 	iftWriteImageP2(imgout,"edges.pgm");
// 
// 
// 	iftDestroyDataSet(&Z1);
// 	iftDestroyDataSet(&Z2);
// 	iftDestroyDataSet(&Zl);
	
	fprintf(stderr,"destroying fileNames...\n");
	destroyFileNames(files);
	destroyFileNames(filesGT);
	
	fprintf(stderr,"destroying char*...\n");
//	free(file_config);
//	free(file_imageNames);
//	free(file_imageNamesGT);
	free(folder_name);
	free(folder_nameGT);

	fprintf(stderr,"destroying MSConvNetwork...\n");
	iftDestroyMSConvNetwork(&msconvnet);
	
	return 0;
}


/* as far as, just a single image
 	fullPath[0] = 0;
//	#pragma omp parallel for private(fullPath)
	for(i = 1; i < nimages ; i++)
	{
		printf("It. = %d, File %s, class %d\n", i, files[i].filename, files[i].truelabel);
		strcpy(fullPath, folder_name);
		strcat(fullPath, files[i].filename);

		iftImage *img = iftReadImageP5(fullPath);
		Z->sample[i].truelabel = files[i].truelabel;
		iftFeatures* features = iftExtractDeepFeaturesMultiScale(mimg, msconvnet);
		if(Z->nfeats != features->n)
			printf("Invalid number of features: %s",files[i].filename);

		//Frees already allocated memory for sample
		free(Z->sample[i].feat);
		Z->sample[i].feat = features->val;

		fullPath[0] = 0;

	}
*/

// 	tic = iftTic();
// 	iftWriteOPFDataSet(Zl,"dataset.data");
// 	toc = iftToc();
// 	printf("Dataset saving: %f\n", iftCompTime(tic,toc));
	



// 	tic = iftTic();
// 	iftDataSet* Zedges = iftMImageToLabeledEdgesDataSet(mimg,imgGT,imgWG);
// 	toc = iftToc();
// 	printf("Dataset creation: %f\n", iftCompTime(tic,toc));
// 
// 	tic = iftTic();	
// 	Z1 = iftExtractClass(Zedges,1);
// 	printf("class 1: %d\n",Z1->nsamples);
// 	
// 	Z2 = iftExtractClass(Zedges,2);
// 	printf("class 2: %d\n",Z2->nsamples);
// 	
// 	float fData = 2.; // (float)Z1->nsamples/Z2->nsamples; // 20.;	
// 
// 	int selected = iftSelectSupTrainSamples(Z1,(float)Z2->nsamples/Z1->nsamples*fData);
// 	printf("class 1 selected: %d\n",selected);
// 
// 	iftDestroyDataSet(&Zedges);
// 	Zedges = iftCreateDataSet(Z2->nsamples+selected,Z2->nfeats);
// 	Zedges->nclasses = 2;
// 	for( i = 0; i < Z2->nsamples; i++)
// 	{
// 		Zedges->sample[i].truelabel = Z2->sample[i].truelabel;
// 		memcpy(Zedges->sample[i].feat,(Z2->sample[i].feat),sizeof(float)*Zedges->nfeats);
// 	}
// 	int j;
// 	for( i = 0,j = 0; i < Z1->nsamples; i++)
// 	{
// 		if (Z1->sample[i].status == IFT_TRAIN)
// 		{	Zedges->sample[Z2->nsamples+j].truelabel = Z1->sample[i].truelabel;
// 			memcpy(Zedges->sample[Z2->nsamples+j].feat,(Z1->sample[i].feat),sizeof(float)*Zedges->nfeats); j++;}
// 	}
// 
// 	selected = iftSelectSupTrainSamples(Zedges,(float)1.0);
// 	printf("for final dataset - selected: %d\n",selected);
// 	Zl = iftCreateDataSet(selected,Zedges->nfeats);
// 	Zl->nclasses = 2;
// 	for( i = 0,j = 0; i < Zedges->nsamples; i++)
// 	{
// 		if (Zedges->sample[i].status == IFT_TRAIN)
// 		{	Zl->sample[j].truelabel = Zedges->sample[i].truelabel;
// 			memcpy(Zl->sample[j].feat,(Zedges->sample[i].feat),sizeof(float)*Zedges->nfeats); j++;}
// 	}
// 
// 	// Reset status
// //	iftSetStatus(Zedges, IFT_TEST);
// 	toc = iftToc();
// 	printf("Dataset reduction/selection(%3.2f): %f\n", fData,iftCompTime(tic,toc));
