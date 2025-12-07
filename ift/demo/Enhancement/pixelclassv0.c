#include "ift.h"
#include "iftTrainEdges.h"
#include "common.h"

//Parse the config file to create an convolutional network
iftMSConvNetwork* parseConfigFile (char * file_config)
{
	FILE *fp = fopen(file_config, "r");
	if (!fp) iftError("File not found", "msconvnet - parseConfigFile");
	char line[512];
	char *pch;

	int nlayers=-1;
	int bConv=1;
	int bPool=1;
	int bNorm=1;
	int nscales=1;
	float scale_ratio = 2;
	int input_bands=1;
	int input_norm_ws=3;
	int output_norm_ws=3;
	int *nkernels = 0;
	int *kernel_ws = 0;
	int *pooling_ws = 0;
	int *stride = 0;
	float *alpha = 0;
	int *normalization_ws=0;

	int i=0, j=0;
	char* rd=NULL;
	while (!feof(fp))
	{
		j=0;
		rd=fgets(line, sizeof(char)*512, fp);
		if (rd == NULL) iftError("error while reading from the config file", "msconvnet - parseConfigFile");
		
		if (i >0 && nlayers == -1) iftError("N_LAYERS must be the first parameter on the config file", "msconvnet - parseConfigFile");
		pch = strtok (line,", ");
		if (strcmp(pch, "N_LAYERS")==0)
		{
			pch = strtok (NULL,", ");
			nlayers = atoi(pch);
			nkernels = iftAllocIntArray(nlayers);
			kernel_ws = iftAllocIntArray(nlayers);
			pooling_ws = iftAllocIntArray(nlayers);
			stride = iftAllocIntArray(nlayers);
			alpha = iftAllocFloatArray(nlayers);
			normalization_ws = iftAllocIntArray(nlayers);
		}
		else if (strcmp(pch, "NO_CONVOLUTION")==0)
		{
			bConv = 0;   
		}
		else if (strcmp(pch, "NO_POOLING")==0)
		{
			bPool = 0;   
		}
		else if (strcmp(pch, "NO_NORMALIZATION")==0)
		{
			bNorm = 0;   
		}
		else if (strcmp(pch, "N_SCALES")==0)
		{
			pch = strtok (NULL,", ");
			nscales = atoi(pch);
		}
		else if (strcmp(pch, "SCALE_RATIO")==0)
		{
			pch = strtok (NULL,", ");
			scale_ratio = (float)atof(pch);
		}
		else if (strcmp(pch, "N_BANDS_INPUT")==0)
		{
			pch = strtok (NULL,", ");
			input_bands = atoi(pch);
		}
		else if (strcmp(pch, "INPUT_NORM_SIZE")==0)
		{
			pch = strtok (NULL,", ");
			input_norm_ws = atoi(pch);
		}
		else if (strcmp(pch, "OUTPUT_NORM_SIZE")==0)
		{
			pch = strtok (NULL,", ");
			output_norm_ws = atoi(pch);
		}
		else if (strcmp(pch, "N_KERNELS")==0)
		{
			while (j < nlayers)
			{
//				printf("%s ", pch);
				pch = strtok (NULL,", ");
				nkernels[j] = atoi(pch);
				j++;
			}
		}
		else if (strcmp(pch, "SIZE_KERNELS")==0)
		{
			while (j < nlayers)
			{
				pch = strtok (NULL,", ");
				kernel_ws[j] = atoi(pch);
				j++;
			}
		}
		else if (strcmp(pch, "SIZE_POOLING")==0)
		{
			while (j < nlayers)
			{
				pch = strtok (NULL,", ");
				pooling_ws[j] = atoi(pch);
				j++;
			}
		}
		else if (strcmp(pch, "STRIDE")==0)
		{
			while (j < nlayers)
			{
				pch = strtok (NULL,", ");
				stride[j] = atoi(pch);
				j++;
			}
		}
		else if (strcmp(pch, "ALPHA")==0)
		{
			while (j < nlayers)
			{
				pch = strtok (NULL,", ");
				alpha[j] = atoi(pch);
				j++;
			}
		}
		else if (strcmp(pch, "SIZE_NORM")==0)
		{
			while (j < nlayers)
			{
				pch = strtok (NULL,", ");
				normalization_ws[j] = atoi(pch);
				j++;
			}
		}

		line[0] = 0;
		i++;
	}
	fclose(fp);
	
	if (!bConv)
	{
		if (nkernels        )	free(nkernels        );	nkernels         = 0;
		if (kernel_ws       )	free(kernel_ws       );	kernel_ws        = 0;
	}
	if (!bPool)
	{
		if (pooling_ws      )	free(pooling_ws      );	pooling_ws       = 0;
		if (stride          )	free(stride          );	stride           = 0;
		if (alpha           )	free(alpha           );	alpha            = 0;
	}
	if (!bNorm)
	{
		if (normalization_ws)	free(normalization_ws);	normalization_ws = 0;
	}

	printf("Number of Layers: %d\n", nlayers);
	printf("Number of Scales: %d\n", nscales);
	printf("Scale ratio: %.2f\n", scale_ratio);
	printf("Number of Bands: %d\n", input_bands);
	printf("Input Norm. W.Size: %d\n", input_norm_ws);
	if (bConv)
	{
		printf("Number of Kernels: ");
		for (i=0; i < nlayers; i++)
		{
			printf("%d ", nkernels[i]);	
		}
		
		printf("\nSize of Kernels: ");
		for (i=0; i < nlayers; i++)
		{
			printf("%d ", kernel_ws[i]);	
		}
	}
	if (bPool)
	{
		printf("\nSize of Pooling: ");
		for (i=0; i < nlayers; i++)
		{
			printf("%d ", pooling_ws[i]);	
		}
		printf("\nStride: ");
		for (i=0; i < nlayers; i++)
		{
			printf("%d ", stride[i]);	
		}
		printf("\nAlpha: ");
		for (i=0; i < nlayers; i++)
		{
			printf("%.2f ", alpha[i]);
		}
	}
	if (bNorm)
	{
		printf("\nNorm. W.Size: ");
		for (i=0; i < nlayers; i++)
		{
			printf("%d ", normalization_ws[i]);
		}
	}
	printf("\nOutput Norm. W.Size: %d", output_norm_ws);
	printf("\n--------------\n");
	return iftCreateMSConvNetwork(nlayers, nscales, scale_ratio, input_bands,input_norm_ws, nkernels,kernel_ws,pooling_ws,stride,alpha,normalization_ws,output_norm_ws);
}

iftDataSet* iftMImageToLabeledPixelsDataSet(iftMImage* mimg,iftImage* imgGT, iftImage* imglabels,iftAdjRel* A,int nsamples,float percBorder)
{ // 2d iftMImage
	fprintf(stderr,"pixelsize: %d\n",nsamples);
	  
	//iftVerifyImageDomains(imgGT,imglabels,"parseConfigFile");
	
	iftDataSet* Zedges = iftCreateDataSet(nsamples, mimg->m);
	Zedges->nclasses = 2;
	
	// generating vertical edges
	int s,p,q,b,numB,numNB,countB,countNB;
	numB = (int)(percBorder*nsamples);
	numNB = nsamples - numB;
	printf("numB: %d,numNB: %d, nsamples: %d\n", numB, numNB, nsamples);
	countB = 0;
	countNB = 0;
	s = 0;
	int* selectedPIX = iftAllocIntArray(mimg->n);
	//for(s = 0, p = 0; p < mimg->n; p++)
	while(s < nsamples)
	{
		p = iftRandomInteger(0,mimg->n-1);
		if (selectedPIX[p] == 0){
			selectedPIX[p] = 1;
			iftVoxel pixel   = iftMGetVoxelCoord(mimg ,p);			
			if ( countB < numB && imgGT->val[p] == 255 )
			{
				Zedges->sample[s].id = s;
				Zedges->sample[s].truelabel = 2;  // border
				for(b = 0; b < mimg->m ; b++)
					Zedges->sample[s].feat[b] = mimg->val[p][b];
				s++;
				countB++;
				//printf("selected 2 p: %d\n",p);
			}
			else
			{ 
				if ( countNB < numNB && imglabels->val[p] == 0 )   
				{ 
					Zedges->sample[s].id = s;
					Zedges->sample[s].truelabel = 1; // non border
					for(b = 0; b < mimg->m ; b++)
						Zedges->sample[s].feat[b] = mimg->val[p][b];
					s++;
					countNB++;
					//printf("selected 1 p: %d\n",p);
				}
			}
		}
	}
	printf("Zedges created \n");
	Zedges->nsamples = nsamples; // gambiarra

	return Zedges;
}

int main(int argc, char **argv) 
{
	
	// read input files
	if (argc != 7)	iftError("Please provide the following parameters:\n <CONFIG_FILE> <FILE_IMAGE_NAMES> <FILE_IMAGE_NAMES_GT> <SAMPLES_BY_IMAGE> <PER_BORDER> <N_THREADS>\n\n", "main");
	char *file_config;
	char *file_imageNames;
	char *file_imageNamesGT;
    int nsamples;
	float percBorder;
    timer *tic, *toc;
        
    file_config 	  = argv[1];
	file_imageNames   = argv[2];
	file_imageNamesGT = argv[3];
    nsamples   = atoi(argv[4]);
	percBorder     = atof(argv[5]);

	//FIXME
	omp_set_num_threads(atoi(argv[6]));

    tic = iftTic();

    // filename image variables
    int idxImg,s;
    int nimages,nimagesGT = 0;
	char *folder_name = iftAllocCharArray(256);
	char *folder_nameGT = iftAllocCharArray(256);
	char fullPath[128];
	FileNames * files = NULL;
	FileNames * filesGT = NULL;
    // read filename images 
    folder_name = getFolderName(file_imageNames);
	nimages = countImages(file_imageNames);
	files = createFileNames(nimages);
	loadFileNames(files, file_imageNames);
	// read filename GT images
	folder_nameGT = getFolderName(file_imageNamesGT);
	nimagesGT = countImages(file_imageNamesGT);
	filesGT = createFileNames(nimagesGT);
	loadFileNames(filesGT, file_imageNamesGT);

	// Show image information
	printf("Folder: %s\n", folder_name);
	printf("Number of Images: %d\n", nimages);
	printf("File with image names: %s\n"   , file_imageNames);
	printf("File with GT image names: %s\n", file_imageNamesGT);
	printf("Number of samples: %d (Border : %6.2f%%)\n",nsamples,percBorder);	
	//printf("Spatial radius: %.4f\n",radius);	
	puts("");

	//FIXME
	iftMSConvNetwork *msconvnet = parseConfigFile(file_config);


	for(idxImg = 0; idxImg < nimages ; idxImg++)
	{

		
		timer* t1, *t2;
		
		t1 = iftTic();

		// read image		
		fullPath[0] = 0;
		strcpy(fullPath, folder_name);
		strcat(fullPath, files[idxImg].filename);
		printf("Processing: (%d/%d) %s/%s\n", idxImg+1,nimages,files[idxImg].filename,filesGT[idxImg].filename);

		fprintf(stderr,"%s\n",fullPath);
		timer *tic, *toc;
		tic = iftTic();	
		iftImage *img = iftReadImageP5(fullPath);

		// read GT image
		fullPath[0] = 0;
		strcpy(fullPath, folder_nameGT);
		strcat(fullPath, filesGT[idxImg].filename);
		iftImage *imgGT = iftReadImageP5(fullPath);

		// ImageToMImage
		iftMImage* imimg = iftImageToMImage(img,YCbCr_CSPACE);
		// Extract Deep Features Multiscale
		iftMImage*  mimg  = iftExtractDeepFeaturesMultiScale(imimg, msconvnet);
		iftImage*  imgLabels;

		iftAdjRel *adjDilate;
		float radiusDilate = 2;
		adjDilate = iftCircularEdges(radiusDilate);
		imgLabels = iftDilate(imgGT, adjDilate);
		iftWriteImageP2(imgLabels,"dilate.pgm");

		// Create adjacency relation
		float radius = 1.0;
		iftAdjRel* A;
		if (mimg->zsize == 1)
			A = iftCircularEdges(radius);
		else
			A = iftSphericEdges(radius);
		
		// Creating training SET
		tic = iftTic();
		iftDataSet* Zedges = iftMImageToLabeledPixelsDataSet(mimg,imgGT,imgLabels,A,nsamples,percBorder);
		toc = iftToc();
		printf("Dataset creation: %f\n", iftCompTime(tic,toc));

		
		// Training/testing
		iftDataSet      *Z=NULL, *Z1[3], *Z2[3];
		iftSVM          *svm=NULL;
		
		float train_perc;
		int n;
		int num_of_comps;
		int reduction;
		int kernel_type;
		int i;
		float stdev, mean;
		
		Z = Zedges;
		train_perc = 0.9;
		n    = 20; 
		reduction    = 0;
		num_of_comps = 0;
		kernel_type = 0;
		float* acc  = iftAllocFloatArray(n); 

		t1 = iftTic();
		float C = 100000;
		float sigma = 0.1;
		svm = iftCreateLinearSVC(C);
		for (i=0; i < 3; i++){
		  Z1[i]=Z2[i]=NULL;
		}
		


		for (i=0; i < n; i++) {    
		  if (Z1[2] != NULL) iftDestroyDataSet(&Z1[2]);
		  if (Z2[2] != NULL) iftDestroyDataSet(&Z2[2]);
		  
		  iftSelectSupTrainSamples(Z,train_perc); // Select training samples
	      
		  Z1[0] = iftExtractSamples(Z,IFT_TRAIN);
	      Z2[0] = iftExtractSamples(Z,IFT_TEST);
		  
		  // no reduction
		  Z1[2] = iftNormalizeDataSet(Z1[0]);
		  iftDestroyDataSet(&Z1[0]);
		  Z2[2] = iftNormalizeTestDataSet(Z1[2],Z2[0]);
		  iftDestroyDataSet(&Z2[0]);

		  iftSVMTrainOVA(svm, Z1[2]); // Training
		  printf("nsamples 1: %d \n",Z1[2]->nsamples);
	      iftSVMClassifyOVA(svm, Z2[2], IFT_TEST); // Classification
		  printf("nsamples 2: %d \n",Z2[2]->nsamples);
		  
		  acc[i] = iftSkewedTruePositives(Z2[2]); // Compute accuracy on test set
		  printf("acc : %f \n",acc[i]);

		}
		int it;
		mean = 0.0;
		for (it=0; it < n; it++)
			mean += acc[it];
		mean /= n;

		stdev = 0.0;
		for (it=0; it < n; it++)
		  stdev += (acc[it]-mean)*(acc[it]-mean);
		if (n > 1)
		  stdev = sqrtf(stdev/(n-1));

		//free(acc); 

		t2 = iftToc();
		fprintf(stdout,"Accuracy of classification is mean=%f, stdev=%f\n",mean,stdev); 
		

		/*
		iftDataSet *Z1,*Z2,*Zl;
		tic = iftTic();

		Z1 = iftExtractClass(Zedges,1);
		printf("class 1: %d\n",Z1->nsamples);
		
		Z2 = iftExtractClass(Zedges,2);
		printf("class 2: %d\n",Z2->nsamples);

		float fData = 2.; // (float)Z2->nsamples/Z1->nsamples; // 20.;	

		int selected = iftSelectSupTrainSamples(Z1,0.9);
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
		Zedges = iftCopyDataSet(Zl, true);
		
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
		printf("min : %d \n",iftFMaximumValue(imgedges));
		printf("max : %d \n",iftFMinimumValue(imgedges));
		printf("samples : %d \n",Ztest[1]->nsamples);
		//iftWriteFImage(imgedges,"edgesF.pgm");
		iftImage* imgout = iftFImageToImage(imgedges, 4095);
		iftDestroyDataSet(&Ztest[1]);
		iftDestroyAdjRel(&A);
	

		iftWriteImageP2(imgout,"edges.pgm");
  
		t2 = iftToc();

		puts("");
		printf("Finished. Time: %f seg\n",iftCompTime(t1,t2)/1000);

		iftDestroyImage(&img);
		iftDestroyImage(&imgGT);
		//iftDestroyImage(&imgWG);
		iftDestroyMImage(&mimg);
		iftDestroyMImage(&imimg);
		*/


	}


	
	return 0;
}
