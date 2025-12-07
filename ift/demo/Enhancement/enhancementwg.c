#include "common.h"
#include "ift.h"
#include "iftTrainEdges.h"

#include <libgen.h>

/*
iftDataSet* iftMImageToEdgesDataSet(iftMImage* mimg, ,iftAdjRel* A)
{ 	// 2d iftMImage
	fprintf(stderr,"edges: %d, pixelsize: %d\n",(A->n-1)*mimg->n,mimg->m);
	
	iftDataSet* Zedges = iftCreateDataSet((A->n-1)*mimg->n, mimg->m);
	Zedges->nclasses = 1;
	
	// generating vertical edges
	int s,p,q,b;
	for(s = 0, p = 0; p < mimg->n; p++)
	{
		iftVoxel pixel   = iftMGetVoxelCoord(mimg ,p);

		int i;
		for(i = 1; i < A->n; i++)
		{
			iftVoxel neighbor   = iftGetAdjacentVoxel(A,pixel  ,i);

			if (iftMValidVoxel(mimg,neighbor))
				q   = iftMGetVoxelIndex(mimg ,neighbor);

			else
				q   = p;

			if ( (imgGT->val[p] != imgGT->val[q]) && (imglabels->val[p] != imglabels->val[q]) )
			{
				Zedges->sample[s].id = s;
				Zedges->sample[s].truelabel = 2;  // edges
				for(b = 0; b < mimg->m ; b++)
					Zedges->sample[s].feat[b] = fabs(mimg->val[p][b] - mimg->val[q][b]);
				s++;
			}
			else
			{ 
				if ( (imglabels->val[p] == imglabels->val[q]) && (imgGT->val[p] == imgGT->val[q]) )   
				{ 
					Zedges->sample[s].id = s;
					Zedges->sample[s].truelabel = 1; // non edges
					for(b = 0; b < mimg->m ; b++)
						Zedges->sample[s].feat[b] = fabs(mimg->val[p][b] - mimg->val[q][b]);
					s++;
				}
			}
		}
	}
	Zedges->nsamples = s; // gambiarra

	return Zedges;
}
*/
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
			Zedges->sample[offsetSample+s].feat[b] = mimg->val[p][b] * mimg->val[q][b];
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
	iftDataSet      *Z=NULL, *Z1[3], *Z2[3];
	iftSVM          *svm=NULL;
	
	float train_perc;
	int n;
	int i;
	float stdev, mean;
	
	Z = Zedges;
	train_perc = 0.9;
	n    = 10; 
	float* acc  = iftAllocFloatArray(n); 

	t1 = iftTic();
	float C = 1;
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

	/*
	// write image
	iftImage *imgtest = iftReadImageP5("test.pgm");
	iftDataSet* Zt = iftMImageToLabeledEdgesDataSet(mimg,imgGT,imgWG,A);
	iftSVMClassifyOVO(svm, Zt, IFT_TEST);
	iftFImage* imgedges = iftEdgesDataSetToFImage(Zt,mimg,A);
	printf("min : %d \n",iftFMaximumValue(imgedges));
	printf("max : %d \n",iftFMinimumValue(imgedges));
	printf("samples : %d \n",Ztest[1]->nsamples);
	iftImage* imgout = iftFImageToImage(imgedges, 4095);
	iftWriteImageP2(imgout,"edges.pgm");
	*/
	
	//iftDestroySVM(svm);
	//iftDestroyDataSet(&Z);


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
	
	tfinish = iftToc();
	printf("total classification by SVM phase: %f seg\n",iftCompTime(tstart,tfinish)/1000);

	return 0;
}

/*
		Creating imgWG
		float spatial_radius = 1.5;
		float volume_threshold = 2000.;
		iftAdjRel* adj_relation = iftCircular(spatial_radius);
		iftImage* basins = iftImageBasins(img,adj_relation);
		
		int nregions = -1;
		iftImage* marker=NULL; iftImage* imgWG=NULL;
		while(nregions < 0)
		{
			if (marker != NULL) iftDestroyImage(&marker);
			marker = iftVolumeClose(basins,volume_threshold);
			if (imgWG  != NULL) iftDestroyImage(&imgWG);
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

		fprintf(stderr,"(%d,%d) : (%d,%d)\n",img->xsize,img->ysize,imgWG->xsize,imgWG->ysize);
		iftVerifyImageDomains(img,imgWG,"main");
		
*/
