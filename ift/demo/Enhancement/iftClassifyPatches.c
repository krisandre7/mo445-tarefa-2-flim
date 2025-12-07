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

iftFImage* iftPixelDataSetToFImage_Local(iftDataSet* dataset,iftMImage* mimg)
{
	//printf("ini iftPixelDataSetToFImage \n");
	//printf("img pixels %d\n", mimg->n);
	//printf("nsamples %d\n", dataset->nsamples);
	fprintf(stderr,"(%d x %d):\n",mimg->n,dataset->nsamples);
	if (mimg->n != dataset->nsamples)
		iftError("incompatible dimensions ", "iftEdgesDataSetToImage");
	
	iftFImage* output = iftCreateFImage(mimg->xsize,mimg->ysize,mimg->zsize);
	
	int s,countB,countNB;
	countB = 0;
	countNB = 0;
	for(s = 0;s < dataset->nsamples; s++)
	{
		if (dataset->sample[s].label == 2) // edges
		{
			if (dataset->sample[s].weight > 0)
			{
				output->val[s] = dataset->sample[s].weight;
			}
			countB++;
		}else{
			countNB++;
		}
	}
	printf("Border: %d, Non Border: %d \n",countB,countNB);
	return output;
}

iftImage *iftBorderImageByPatches(iftImage *img, float volume_threshold, iftDataSet *Ztrain, float cSVM, iftSVM *svm, iftCplGraph *graph, iftConvNetwork* convnet){
	iftImage *basins,*marker,*label_image,*img_out,*border_wg;
	iftMImage *input, *output;
	iftDataSet *Ztest[2], *Z;
	int p,i,j,q,count,numBorders,center,r,ibr;
	iftVoxel u,v;
	iftAdjRel *APatch,*AAux;
	iftMatrix *borderRegions;
	iftAdjRel *A = iftCircular(sqrtf(2.0));
	iftFeatures *all_features;
	img_out = iftCreateImage(img->xsize,img->ysize,img->zsize);
	int xsize,ysize,num_scales;
	num_scales = 3;
	int scales[num_scales];

	scales[0] = 7;
	scales[1] = 15;
	scales[2] = 25;

	xsize = scales[num_scales-1];
	ysize = scales[num_scales-1];


	// Get Watergray
	basins = iftImageBasins(img,A);
	marker = iftVolumeClose(basins,volume_threshold);
	label_image = iftWaterGray(basins,marker,A);
	border_wg = iftWaterGrayBorder(label_image,A);

	iftWriteImageP2(label_image,"wg.pgm");
	iftWriteImageP2(border_wg,"wgb.pgm");
	/*
	for(p=0;p<label_image->n;p++){
		if(p > (label_image->n-100))
			printf("p: %d\n",label_image->val[p]);
	}
	*/

	// counting border regions
	numBorders = 0;
	int* i1 = iftAllocIntArray(100000);
	int* i2 = iftAllocIntArray(100000);
	int* pos = iftAllocIntArray(100000);
	iftMatrix *regions = iftCreateMatrix(1000,1000);
	for (p=0; p < label_image->n; p++) {
		u.x = iftGetXCoord(label_image,p);
		u.y = iftGetYCoord(label_image,p);
		u.z = iftGetZCoord(label_image,p);
		for (j=0; j < A->n; j++) {
		  v.x = u.x + A->dx[j];
		  v.y = u.y + A->dy[j];
		  v.z = u.z + A->dz[j];
		  if (iftValidVoxel(label_image,v)){
		q = iftGetVoxelIndex(label_image,v);
		if (label_image->val[p] < label_image->val[q]){
		  regions->val[iftGetMatrixIndex(regions,label_image->val[p],label_image->val[q])]++;
		  i1[numBorders] = label_image->val[p];
		  i2[numBorders] = label_image->val[q];
		  pos[numBorders] = p;
		  numBorders++;
		  break;
		}
		  }
		}
	}

	// Create indices
	int* ind1 = iftAllocIntArray(numBorders);
	int* ind2 = iftAllocIntArray(numBorders);
	int* position = iftAllocIntArray(numBorders);
	printf("count: %d \n",numBorders);
	for(i=0;i<numBorders;i++){
		ind1[i] = i1[i];
		ind2[i] = i2[i];
		position[i] = pos[i];
	}
	free(i1);
	free(i2);
	free(pos);

	// Get number of regions borders (transitions)
	int num_border_regions = 0;
	int maxBorders = 0;
	for(i=0;i<1000;i++){
		for(j=0;j<1000;j++){
			if(regions->val[iftGetMatrixIndex(regions,i,j)] > 0){
				//printf("border regions: %d, %d, %d ",i,j,(int)regions->val[iftGetMatrixIndex(regions,i,j)]);
				if( regions->val[iftGetMatrixIndex(regions,i,j)] > maxBorders)
					maxBorders = regions->val[iftGetMatrixIndex(regions,i,j)];
				num_border_regions++;
			}
		}
	}
	printf("num tansitions regions: %d , max borders : %d, numBorders: %d  \n",num_border_regions, maxBorders, numBorders);

	// Set index
	int* reg1 = iftAllocIntArray(num_border_regions);
	int* reg2 = iftAllocIntArray(num_border_regions);
	count = 0;
	for(i=0;i<1000;i++){
		for(j=0;j<1000;j++){
			if(regions->val[iftGetMatrixIndex(regions,i,j)] > 0){
				reg1[count] = i;
				reg2[count] = j;
				count++;
			}
		}
	}

	// Create dataset
	int xsize_arr[convnet->nstages], ysize_arr[convnet->nstages], zsize_arr[convnet->nstages], nbands[convnet->nstages];
	iftImageDimensionsAlongNetwork(convnet, xsize_arr, ysize_arr, zsize_arr, nbands);
	int nfeatures = xsize_arr[convnet->nstages - 1] * ysize_arr[convnet->nstages - 1] *
			zsize_arr[convnet->nstages - 1] * nbands[convnet->nstages - 1];
	Z = iftCreateDataSet(num_border_regions, nfeatures*num_scales);
	printf("Z samples: %d, features :%d\n",Z->nsamples,Z->nfeats);

	iftMatrix *pxbRegions = iftCreateMatrix(num_border_regions,maxBorders);
	// Get Regions
	for(ibr=0;ibr<num_border_regions;ibr++){
		int* pixels = iftAllocIntArray(maxBorders);
		for(j=0;j<maxBorders;j++){
			pixels[j] = -1;
		}
		count = 0;
		for(j=0;j<numBorders;j++){
			if(ind1[j]==reg1[ibr] && ind2[j]==reg2[ibr]){
				pixels[count] = position[j];
				pxbRegions->val[iftGetMatrixIndex(pxbRegions,ibr,count)] = (float)position[j];
				//img_out->val[pixels[count]] = 255;
				count++;
				//printf("position %d : %d\n",i,position[j]);
			}
		}
		//int pos_med = (int)pxbRegions->val[iftGetMatrixIndex(pxbRegions,ibr,(count/2))];
		//img_out->val[pos_med] = 255;
		//img_out->val[pixels[count/2]] = 255;
		//iftWriteImageP2(img_out,"wg12.pgm");
		p = pixels[count/2];
		APatch = iftRectangular(xsize,ysize);
		iftVoxel pixel   = iftGetVoxelCoord(img ,p);
		for(i=1;i<APatch->n;i++){
			iftVoxel neighbor   = iftGetAdjacentVoxel(APatch,pixel,i);
			if (!iftValidVoxel(img,neighbor)){
				break;
			}
		}
		if(i==APatch->n){ // valid Patch

			for(int is=0;is<num_scales;is++){
				if(num_scales > 1){
					xsize = scales[is];
					ysize = scales[is];
				}
				iftImage *patch = iftCreateImage(xsize,ysize,1);
				if (img->Cb != NULL) {
					patch->Cb = iftAllocUShortArray(patch->n);
					patch->Cr = iftAllocUShortArray(patch->n);
				}
				center = (patch->xsize * patch->ysize) /2;
				iftVoxel vox_center   = iftGetVoxelCoord(patch ,center);
				AAux = iftRectangular(patch->xsize,patch->ysize);
				iftCopyVoxelSize(img,patch);
				patch->val[center] = img->val[p];  // Do for color image
				if (img->Cb != NULL) {
					patch->Cb[center]=img->Cb[p];
					patch->Cr[center]=img->Cr[p];
				}
				for(i=1;i<AAux->n;i++){
					iftVoxel u   = iftGetAdjacentVoxel(AAux,pixel,i);
					q = iftGetVoxelIndex(img,u);
					iftVoxel v   = iftGetAdjacentVoxel(AAux,vox_center,i);
					r = iftGetVoxelIndex(patch,v);
					patch->val[r] = img->val[q];
					if (img->Cb != NULL) {
						patch->Cb[r]=img->Cb[q];
						patch->Cr[r]=img->Cr[q];
					}
				}
				iftImage* img_patch;
				if (img->Cb != NULL) {
					img_patch = iftInterp2D(patch, (float)convnet->input_xsize/(float)patch->xsize, (float)convnet->input_ysize/(float)patch->ysize);
				}else{
					img_patch = iftInterp2D(patch, (float)convnet->input_xsize/(float)patch->xsize, (float)convnet->input_ysize/(float)patch->ysize);
				}
				iftDestroyImage(&patch);

				input  = iftImageToMImage(img_patch, RGB_CSPACE);
				iftDestroyImage(&img_patch);

				output = iftApplyConvNetwork(input, convnet);
				iftDestroyMImage(&input);

				if(is==0) // Create new features
					all_features = iftCreateFeatures(num_scales*output->n*output->m);
				printf("values %d, %d, %d, base: %d \n",num_scales,output->n,output->m, (is*output->n*output->m));

				// Add features
				int p,b;
				for(b = 0; b < output->m; b++){
					for(p = 0; p < output->n; p++){
						all_features->val[(is*output->n*output->m) + p + b*output->n] = output->val[p][b];
					}
				}
			}
			printf("added features \n");
			// Set features
			//Z->sample[ibr].truelabel = img_names[ibr].attribute;
			free(Z->sample[ibr].feat);
			Z->sample[ibr].feat = all_features->val;
			printf("Z set features \n");
			iftDestroyMImage(&output);
		}
		printf("end ibr %d \n",ibr);


		free(pixels);
	}


	// Show information dataset
	printf("Num samples: %d, Num feats: %d \n",Z->nsamples,Z->nfeats);
	Z->nclasses = 2;
	//for(int i=0;i<Z->nsamples;i++){
	//	Z->sample[i].truelabel = Z->sample[i].truelabel + 1;  // class 1 and 2
	//}

	// Classify
	timer *tic, *toc;
	tic = iftTic();
	Ztest [0] = iftExtractSamples(Z,IFT_TEST);
	Ztest [1] = iftNormalizeTestDataSet(Ztrain,Ztest[0]);
	iftDestroyDataSet(&Ztest [0]);
	toc = iftToc();
	fprintf(stderr,"selecting training(%d)/testing(%d) data: %f\n",Ztrain->nsamples,Ztest[1]->nsamples,iftCompTime(tic,toc));


	tic = iftTic();
	if (cSVM != 0.0)
		iftSVMLinearClassifyOVA(svm, Ztest[1], IFT_TEST); // Classification
	else
		iftClassify(graph,Ztest[1]);                  // Classify test samples
	toc = iftToc();
	fprintf(stderr,"classifying the test set: %f\n", iftCompTime(tic,toc));


	// Show classified pixels
	int s,countB,countNB,pos_pix;
	countB = 0;
	countNB = 0;
	for(s = 0;s < Ztest[1]->nsamples; s++){
		printf("for  %d ",Ztest[1]->sample[s].label);
		if (Ztest[1]->sample[s].label > (Ztrain->nclasses/2)) // edges
		{
			//printf("label == 2  %d ",s);
			if (Ztest[1]->sample[s].weight > 0)
			{
				//printf("label >0  %d \n",s);
				for(j = 0;j < maxBorders; j++){
					if(pxbRegions->val[iftGetMatrixIndex(pxbRegions,s,j)] >= 0.0){
						pos_pix = (int)pxbRegions->val[iftGetMatrixIndex(pxbRegions,s,j)];
						//img_out->val[pos_pix] = 255;
						img_out->val[pos_pix] = Ztest[1]->sample[s].weight;
					}
				}
			}
			countB++;
		}else{
			countNB++;
		}
	}
	printf("B %d,NB %d\n ",countB, countNB);

	iftDestroyDataSet(&Ztest[1]);
	iftDestroyMatrix(&regions);
	iftDestroyMatrix(&pxbRegions);
	free(ind1);
	free(ind2);


	return img_out;
}


int main(int argc, char **argv) 
{
	if (argc != 6)	iftError("Please provide the following parameters:\n<DATASET_PIXELS_FILE> <CONVNET_FILE> <FILE_INPUT_NAMES> <cSVM(0=OPF)> <N_THREADS>\n\n", "main");

	iftSVM      *svm   = NULL;
	iftCplGraph *graph = NULL;

	int idxImg,totalPixels;
	float cSVM;
	char *file_datasetpixels;
	char *file_inputImages;
	char *file_convnet  ;
	float volume_threshold;

	file_datasetpixels = argv[1];
	file_convnet = argv[2];
	file_inputImages   = argv[3];
	cSVM      = atof(argv[4]);
	omp_set_num_threads(atoi(argv[5]));
	iftImage *img_out;
	//volume_threshold = 500;
	volume_threshold = 1000;

	printf("DataSet pixels file: %s\n", file_datasetpixels);	
	puts("");

	iftAdjRel *AdjBasins = iftCircular(sqrtf(2.0));

	printf("Reading TrainPixels file and corresponding input images\n");
	iftDataSet *Zpixels = iftReadOPFDataSet(file_datasetpixels);
	printf("Num samples: %d, Num feats: %d \n",Zpixels->nsamples,Zpixels->nfeats);
	/*
	Zpixels->nclasses = 2;
	for(int i=0;i<Zpixels->nsamples;i++){
		Zpixels->sample[i].truelabel = Zpixels->sample[i].truelabel + 1;  // class 1 and 2
	}
	*/
	
	timer *tic, *toc;
	puts("");puts("");	
	printf("\ntraining and learning classification phase (cSVM:%.2f)\n",cSVM);
	

	iftDataSet *Ztrain[2];
	//float* acc  = iftAllocFloatArray(nRUN); 

	// Read CONVNET
	iftConvNetwork* convnet = iftReadConvNetwork(file_convnet);

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


	//----- Train Classifier
	// Select Train Samples
	iftSelectSupTrainSamples(Zpixels,(float)0.999); // Selecting training samples
	iftSetStatus(Zpixels, IFT_TRAIN);

	Ztrain[0] = iftExtractSamples(Zpixels,IFT_TRAIN);

	Ztrain[1] = iftNormalizeDataSet(Ztrain[0]);
	iftDestroyDataSet(&Ztrain[0]);

	tic = iftTic();
	if (cSVM != 0.0) {
		svm=iftCreateLinearSVC(cSVM);
		iftSVMTrainOVA(svm, Ztrain[1]); 	// Training
	}else {
		graph = iftCreateCplGraph(Ztrain[1]);    // Training
		iftSupTrain(graph);
	}
	toc = iftToc();
	fprintf(stderr,"training svm classifier: %f\n", iftCompTime(tic,toc));
	//----- End Train Classifier

	// For all input images
	for( idxImg = 0; idxImg < nimages; idxImg++)
	{
		// Read input images
		fullPath[0] = 0;
		strcpy(fullPath, folder_name);
		strcat(fullPath, files[idxImg].filename);
		printf("full path: %s\n",fullPath);
		iftImage *img = iftReadImageP5(fullPath);

		/*
		// Extract Deep Features
		iftMImage *mimg = iftImageToMImage(img,WEIGHTED_YCbCr_CSPACE);
		iftMImage *msimg = iftApplyConvNetwork(mimg, convnet);
		iftDataSet* Zimg = iftMImageToPixelDataSet(msimg);
		*/

		// Code for classify
		img_out = iftBorderImageByPatches(img, volume_threshold, Ztrain[1], cSVM, svm, graph, convnet);
		iftWriteImageP2(img_out,files[idxImg].filename);
		/*
		// write labeled image
		iftFImage* imgedges = iftPixelDataSetToFImage(Ztest[1],mimg);
		iftImage* imgout = iftFImageToImage(imgedges, 4095);
		*/
		
	  	fullPath[0] = 0;
	  	strcpy(fullPath, folder_name);
	  	strcat(fullPath, files[idxImg].filename);
	  	strcat(fullPath, ".lbl.pgm");

		//iftWriteImageP2(imgout,fullPath);

		//iftDestroyDataSet(&Ztest[1]);


	}

	iftDestroyDataSet(&Ztrain[1]); // due to OPF usage, train data should be destroy only after test
	if (cSVM != 0.0)
		iftDestroySVM(svm);
	if (graph != NULL)
		iftDestroyCplGraph(&graph);

	iftDestroyDataSet(&Zpixels);


	return 0;
}
