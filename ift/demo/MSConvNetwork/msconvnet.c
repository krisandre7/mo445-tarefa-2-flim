#include "ift.h"


//Structre that stores the name of all image and respective class.
typedef struct ift_filenames { 
  char filename[256];
  int class;
} FileNames;

FileNames * createFileNames (int nimages)
{
	FileNames * ITS = (FileNames *) malloc (nimages * sizeof(FileNames));
	return (ITS);
}
void destroyFileNames (FileNames * ITS)
{
	if (ITS != NULL)
	{
		free(ITS);
		ITS = NULL;
	}
}

//Count the number of images in the file with the image names
int countImages (char * file_imageNames)
{
	char line[512],*rd=NULL;
	FILE *fp = fopen(file_imageNames, "r");
	if (!fp) iftError("File not found", "msconvnet - countImages");
	int nlines = 0;
	while(!feof(fp))
	{
		rd=fgets(line, sizeof(char)*512, fp);
		if (rd == NULL) iftError("error while reading from the config file", "msconvnet - countImages");
		
		if (strlen(line) > 1)
			nlines++;
		line[0] = 0;
	}
	fclose(fp);
	return nlines;
}
//Get image database path based on the file with the image names
char * getFolderName (char * file_imageNames)
{
	int i=0;
	int position = -1;
	char * output = iftAllocCharArray(256);
	for (i =0; i<strlen(file_imageNames); i++)
		if (file_imageNames[i] == '/')
			position = i;
	if (position != -1)
	{
		for (i= 0; i < position; i++)
			output[i] = file_imageNames[i];
		output[i] = 0;
		strcat(output, "/");
		return output;
	}
	else
	{
		iftError("The file with the name and class must be inside the folder with the images", "msconvnet - getFolderName");
		return output;
	}
}
//Load all the image names and classes in the FileNames structure
void loadFileNames (FileNames * FN, char file_imageNames[])
{
	char line[512];
	int i=0;
	char* rd=NULL;
	FILE *fp = fopen(file_imageNames, "r");
	if (!fp) iftError("File not found", "msconvnet - loadFileNames");
	int nimages = countImages(file_imageNames);
	while(!feof(fp) || i < nimages)
	{
		rd=fgets(line, sizeof(char)*512, fp);	
		if (rd == NULL) iftError("error while reading from the config file", ",msconvnet - loadFileNames");
		
		if (strlen(line) <= 1)
			continue;

		strcpy(FN[i].filename, line);
		line[0] = 0;
		if (FN[i].filename[strlen(FN[i].filename)-1] == '\n') // Is there a newline character at the end
			FN[i].filename[strlen(FN[i].filename)-1] = 0;
		i++;
	}
	fclose(fp);
}
//Get the number of classes of all images loaded in the FileNames structure
int compareClasses (  void * a,   void * b)
{
	FileNames *first = (FileNames*) a;
	FileNames *second = (FileNames*) b;
	if ( first->class <  second->class ) return -1;
	if ( first->class == second->class ) return 0;

	return 1;
}

int getNumberClasses (FileNames * FN, int nimages)
{
	qsort(FN, nimages, sizeof(FileNames), compareClasses);
	int distinct = 1;
	int previous = FN[0].truelabel;
	int current;
	for (int i=0; i < nimages; i++)
	{
		current = FN[i].truelabel;
		if (current != previous)
		{
			distinct++;
			previous = current;
		}
	}
	return distinct;
}

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
	puts("");
	

	//FIXME
	iftMSConvNetwork *msconvnet = parseConfigFile(file_config);

//	#pragma omp parallel for private(fullPath)
	for(idxImg = 1; idxImg < nimages ; idxImg++)
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
		float spatial_radius = 1.5;
		float volume_threshold = 2000.;
		iftAdjRel* adj_relation = iftCircular(spatial_radius);
		iftImage* basins = iftImageBasins(img,adj_relation);
		
		int nregions = -1;
		iftImage* marker=NULL; iftImage* imgWG=NULL;
		while(nregions < 0)
		{
	//		if (marker != NULL) iftDestroyImage(&marker);
			marker = iftVolumeClose(basins,volume_threshold);
	//		if (imgWG  != NULL) iftDestroyImage(&imgWG);
			imgWG = iftWaterGray(basins,marker,adj_relation);
			nregions = iftMaximumValue(imgWG);
			printf("volume_threshold: %f, nregions: %d\n",volume_threshold,nregions);
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
		iftMImage* imimg = iftImageToMImage(img);
		iftMImage*  mimg  = iftExtractDeepFeaturesMultiScale(imimg, msconvnet);

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
		// gambiarra for iftSelectSupTrainSamples works
		Z2->nclasses--; for(s=0;s<Z2->nsamples;s++) Z2->sample[s].truelabel--;
		printf("class 2: %d\n",Z2->nsamples);
		
		float fData = 2.; // (float)Z2->nsamples/Z1->nsamples; // 20.;	

		int selected = iftSelectSupTrainSamples(Z2,(float)Z1->nsamples/Z2->nsamples*fData);
		printf("class 2 selected: %d\n",selected);

		iftDestroyDataSet(&Zedges);
		Zedges = iftCreateDataSet(Z1->nsamples+selected,Z1->nfeats);
		Zedges->nclasses = 2;
		for( s = 0; s < Z1->nsamples; s++)
		{
			Zedges->sample[s].truelabel = Z1->sample[s].truelabel;
			memcpy(Zedges->sample[s].feat,(Z1->sample[s].feat),sizeof(float)*Zedges->nfeats);
		}
		int j;
		for( s = 0,j = 0; s < Z2->nsamples; s++)
		{
			if (Z2->sample[s].status == IFT_TRAIN)
			{	Zedges->sample[Z1->nsamples+j].truelabel = Z2->sample[s].truelabel+1;
				memcpy(Zedges->sample[Z1->nsamples+j].feat,(Z2->sample[s].feat),sizeof(float)*Zedges->nfeats); j++;}
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
		
		// Reset status
	//	iftSetStatus(Zedges, IFT_TEST);
		toc = iftToc();
		printf("Dataset reduction/selection(%3.2f): %f\n", fData,iftCompTime(tic,toc));
		
		iftSelectSupTrainSamples(Zl,1.0);
		Zedges = iftNormalizeDataSet(Zl);
		iftDestroyDataSet(&Zl);
		
		fprintf(stderr,"data normalization\n");
		iftDestroyDataSet(&Zedges);
  
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
