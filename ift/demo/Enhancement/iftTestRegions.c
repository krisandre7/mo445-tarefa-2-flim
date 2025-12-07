#include "ift.h"
#include "common.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


float iftMagnitudVector(float *f1, int n){
  float mag=0.0f;

  for (int i=0; i < n; i++)
    mag += (f1[i])*(f1[i]);

  return(sqrtf(mag));
}

float iftSumVector(float *f1, int n){
  float mag=0.0f;

  for (int i=0; i < n; i++)
    mag += (f1[i]);

  return(mag);
}

void iftNormalizeMMKernel(iftMMKernel *K)
{
  int p,b,k;

 for (k=0; k < K->nkernels; k++) {
   float average = 0.0, norm = 0.0;
   for(b = 0; b < K->nbands; b++){
     for(p = 0; p < K->A->n; p++){
      average += K->weight[k][b].val[p];
    }
  }
  average = average/(K->A->n * K->nbands);

  //The norm cannot be computed before we have the average
  for(b = 0; b < K->nbands; b++){
    for(p = 0; p < K->A->n; p++){
      K->weight[k][b].val[p] = K->weight[k][b].val[p] - average;
      norm += K->weight[k][b].val[p]*K->weight[k][b].val[p];
    }
  }
  norm = sqrt(norm);

  if (norm > Epsilon) {
    //The final value cannot be computed before the norm
    for(b = 0; b < K->nbands; b++){
      for(p = 0; p < K->A->n; p++){
	K->weight[k][b].val[p] = K->weight[k][b].val[p]/norm;
      }
    }
  }

 }

}

iftDataSet *iftMSupervoxelsSumToDataSet(iftMImage *mimg, iftImage *label)
{
	iftDataSet *Z;
	int r,s,p,b,nregions=iftMaximumValue(label);
	int *size=iftAllocIntArray(nregions);

	if (iftMinimumValue(label)<=0)
		iftError("Minimum label value must be 1","iftImageCompsToDataSet");

	//iftVerifyImageDomains(mimg,label,"iftMSupervoxelsSumToDataSet");

	Z=iftCreateDataSet(nregions,mimg->m);
	for (p=0; p < mimg->n; p++) {
		s = label->val[p] - 1;
		size[s]++;
		for(b=0; b < mimg->m;b++)
			Z->sample[s].feat[b] += mimg->val[p][b];
	}

	for(r = 0; r < nregions; r++){
		for(b=0; b < mimg->m; b++)
			Z->sample[r].feat[b] = Z->sample[r].feat[b]/size[r];
	}

	free(size);

	//This may seem redundant, but it is useful for splitting the dataset
	for(r = 0; r < nregions; r++){
		Z->sample[r].id = r + 1;
	}


	iftSetDistanceFunction(Z, 1);

	return(Z);
}

int iftSelectKernelsFromWGRegions(iftImage *img,iftImage *gt, float volume_threshold, int xsize, int ysize, int nsamples, iftAdjRel *AAllow, char *image_name, char *directory_output, iftConvNetwork *convnet, int type_desc){
	int* selectedPIX;
	int b,p,q,r,s,center,i,j,count,num_regions,r0,r1,num_pixels;
	iftImage *basins,*marker,*label_image;
	iftMImage *input,*output;
	char filename_output[400];
	char filename[200];
	iftAdjRel *APatch;
	int* pSelectedRegion;
	int* countRegions;
	iftMMKernel* K;
	iftAdjRel *A = iftCircular(sqrtf(2.0));


	// WG
	basins = iftImageBasins(img,A);
	marker = iftVolumeClose(basins,volume_threshold);
	label_image = iftWaterGray(basins,marker,A);

	// count of regions
	num_regions = iftMaximumValue(label_image);

	// Selected pixels of the region
	//r0 = iftRandomInteger(1,num_regions);
	r0 =11;
	num_pixels = 0;
	for(p=0;p<label_image->n;p++){
		if(label_image->val[p] == r0){
			num_pixels++;
		}
	}
	pSelectedRegion = iftAllocIntArray(num_pixels);
	count = 0;
	for(p=0;p<label_image->n;p++){
		if(label_image->val[p] == r0){
			pSelectedRegion[count] = p;
			count++;
		}
	}

	//printf("iftSelectKernelsFromWGRegions\n");
	input  = iftImageToMImage(img,WEIGHTED_YCbCr_CSPACE);
	APatch = iftRectangular(xsize,ysize);

	// Create Kernels
	K = iftCreateMMKernel(APatch,input->m,nsamples);

	// Select patches from selected region
	printf("pSelectedRegion: %d num_pixels: %d \n",r0,num_pixels);
	selectedPIX = iftAllocIntArray(num_pixels);
	count = 0;
	iftImage *notallowed = iftCreateImage(img->xsize,img->ysize,img->zsize);
	s=0;
	while(s < nsamples && count < num_pixels ){
		j = iftRandomInteger(0,num_pixels-1);
		if(selectedPIX[j] == 0){
			selectedPIX[j] = 1;
			p = pSelectedRegion[j];
			iftVoxel pixel   = iftGetVoxelCoord(img ,p);
			for(i=1;i<APatch->n;i++){
				iftVoxel neighbor   = iftGetAdjacentVoxel(APatch,pixel,i);
				if (iftValidVoxel(img,neighbor)){
					q = iftGetVoxelIndex(img,neighbor);
					if(notallowed->val[q] == 255 ) { // If some adjacent is of another region is an invalid patch
						break;
					}
				}else{
					break;
				}
			}
			if(i==APatch->n){
				for (b=0; b < input->m; b++) {
					for (j=0; j < A->n; j++) {
						iftVoxel v   = iftGetAdjacentVoxel(APatch,pixel,j);
						q = iftGetVoxelIndex(img,v);
						K->weight[s][b].val[j] = input->val[q][b];
						j++;
					}
				}
				notallowed->val[p] = 255;
				s++;
			}
			count++;
		}
	}
	printf("samples %d\n",nsamples);
	iftNormalizeMMKernel(K);


	free(selectedPIX);
	free(pSelectedRegion);

	r = 1;

	iftDrawWaterGray(img,label_image,"outputregions/a.ppm");
	while(r <= num_regions){
		//if(r != r0){
			// Selected pixels of the region
			num_pixels = 0;
			for(p=0;p<label_image->n;p++){
				if(label_image->val[p] == r){
					num_pixels++;
				}
			}
			pSelectedRegion = iftAllocIntArray(num_pixels);
			count = 0;
			for(p=0;p<label_image->n;p++){
				if(label_image->val[p] == r){
					pSelectedRegion[count] = p;
					count++;
				}
			}
			// Select min and max
			int xmin,ymin,xmax,ymax;
			xmin = 999999999;
			ymin = 999999999;
			xmax = -1;
			ymax = -1;
			for(i=0;i<num_pixels;i++){
				p = pSelectedRegion[i];
				iftVoxel u   = iftGetVoxelCoord(img ,p);
				if(u.x < xmin){
					xmin = u.x;
				}
				if(u.y < ymin){
					ymin = u.y;
				}
				if(u.x > xmax){
					xmax = u.x;
				}
				if(u.y > ymax){
					ymax = u.y;
				}
			}
			iftImage *imgRegion = iftCreateImage(xmax-xmin+1,ymax-ymin+1,1);
			if (img->Cb != NULL) {
				imgRegion->Cb = iftAllocUShortArray(imgRegion->n);
				imgRegion->Cr = iftAllocUShortArray(imgRegion->n);
				iftColor RGB, YCbCr;
				RGB.val[0] = 0; RGB.val[1] = 0; RGB.val[2] = 0;
				YCbCr = iftRGBtoYCbCr(RGB);
				for(p=0;p<imgRegion->n;p++){
					imgRegion->val[p] = YCbCr.val[0];
					imgRegion->Cb[p] = YCbCr.val[1];
					imgRegion->Cr[p] = YCbCr.val[2];
				}
			}

			for(i=0;i<num_pixels;i++){
				p = pSelectedRegion[i];
				iftVoxel u   = iftGetVoxelCoord(img ,p);
				iftVoxel v;
				v.x = u.x - xmin;
				v.y = u.y - ymin;
				v.z = u.z;
				q = v.x + imgRegion->tby[v.y];
				imgRegion->val[q] = img->val[p];
				if (img->Cb != NULL) {
					imgRegion->Cb[q] = img->Cb[p];
					imgRegion->Cr[q] = img->Cr[p];
				}
			}
			sprintf(filename_output,"outputregions/reg_%d.ppm",r);
			iftWriteImageP6(imgRegion,filename_output);
			iftDestroyImage(&imgRegion);
			free(pSelectedRegion);

			//printf("end r: %d \n",r);
			//for (y = ymin; y < ymax ; y++){
			//    for (x = xmin; x < xmax; x++){
			//    	p = x + img->tby[y] + img->tbz[0];
			//    }
			//}


		//}
		r++;

	}

	// Create image region distance
	iftImage *imgDistance = iftCreateImage(img->xsize,img->ysize,img->zsize);
	iftDataSet *Z;
	if(type_desc == 0){
		Z = iftSupervoxelsToDataSet(img, label_image);
	}else if(type_desc == 1 || type_desc == 2){
		if(type_desc == 2){
			iftImage *prob   = iftUniformProbImage(input);
			iftUnsupLearnKernels(input,prob,convnet,5000,0.01,1);
			iftDestroyImage(&prob);
		}
		output = iftApplyConvNetwork(input,convnet);
		Z = iftMSupervoxelsToDataSet(output, label_image);
	}else if(type_desc == 3){
		iftDestroyMMKernel(&convnet->k_bank[0]);
		convnet->k_bank[0] = K;
		convnet->nkernels[0] = K->nkernels;
		output = iftApplyConvNetwork(input,convnet);
		Z = iftMSupervoxelsToDataSet(output, label_image);
		// write features
		for (i=0; i < convnet->nkernels[convnet->nlayers-1]; i++) {
			iftImage *imgF = iftMImageToImage(output,255,i);
			if (imgF != NULL){
				sprintf(filename,"outputfeat/feat%d.pgm",i);
				iftWriteImageP5(imgF,filename);
				iftDestroyImage(&imgF);
			}
		}
	}

	// create alpha
	float *alpha = iftAllocFloatArray(Z->nfeats);
	for(i=0;i<Z->nfeats;i++){
		alpha[i] = 1;
	}
	if(type_desc == 0){
		alpha[0] = 0.2;
	}
	// get distances
	float max_dist = -1;
	float* distances = iftAllocFloatArray(Z->nsamples);
	for(r=0;r<Z->nsamples;r++){
		if(type_desc == 3){
			distances[r] = iftSumVector(Z->sample[r].feat, Z->nfeats);
		}else{
			distances[r] = iftDistance1(Z->sample[r0-1].feat, Z->sample[r].feat, alpha, Z->nfeats);
		}
		if(distances[r] > max_dist){
			max_dist = distances[r];
		}
	}
	for(r=0;r<Z->nsamples;r++){
		if(type_desc == 3){
			distances[r] = 1 - (distances[r] / max_dist);
		}else{
			distances[r] = distances[r] / max_dist;
		}

	}
	for(p=0;p<imgDistance->n;p++){
		//if(r0 == label_image->val[p]){
			//imgDistance->val[p] = 4095;
		//}else{
			imgDistance->val[p] = (int)(distances[label_image->val[p]-1] * 4095);
		//}
	}
	iftWriteImageP2(imgDistance,"outputregions/distance.pgm");

	iftDestroyDataSet(&Z);



	return num_regions;
}


void iftDrawWaterGray(iftImage *img, iftImage *label_image, char *filename_output){
	iftImage *label = iftCopyImage(label_image);
	iftImage *clon = iftCopyImage(img);
	iftAdjRel *A = iftCircular(sqrtf(2.0));
	iftAdjRel *B = iftCircular(0.0);
	iftColor RGB, YCbCr;
	RGB.val[0] = 0;
	RGB.val[1] = 0;
	RGB.val[2] = 255;
	YCbCr      = iftRGBtoYCbCr(RGB);
	iftDrawBorders(clon,label,A,YCbCr,B);
	iftWriteImageP6(clon, filename_output);
	iftDestroyImage(&label);
	iftDestroyImage(&clon);
}


int main(int argc, char **argv) {
	if (argc != 8)
		iftError("Please provide the following parameters:\n<IMAGE_DIRECTORY> <COR=1 GRAY=0> <SIZE> <VOLUME> <#PATCHES> <CONVNET> <TYPE_DESC>\n\n", "main");

	//char *file_config = argv[1];
	iftConvNetwork *convnet;
	char *directory_path = argv[1];
	char *directory_path_gt;
	int img_color = atoi(argv[2]);
	int size = atoi(argv[3]);
	float volume_threshold = atof(argv[4]);
	//int num_scales = atoi(argv[5]);
	int nsamples = atoi(argv[5]);
	convnet = iftReadConvNetwork(argv[6]);
	int type_desc = atoi(argv[7]);
	int nimages,nimages_gt,xsize,ysize;
	iftImageNames *img_names,*img_names_gt;
	//iftImage *basins,*marker,*label_image;
	iftImage *img,*gt;
	//iftMImage *input, *output;
	char filename[200];
	char filename_gt[200];
	char filename_output[200];
	char image_name[200];
	char directory_output[200];
	char ext_images[3];
	char ext_gt[3];
	char dot_ext_images[4];
	char dot_ext_gt[4];
	int num_image;
	int convnet_size;

	//iftAdjRel *A = iftCircular(sqrtf(2.0));
	iftAdjRel *AAllow = iftRectangular(5,5);
	xsize = size;
	ysize = size;
	num_image = 0;
	convnet_size = 70;

	omp_set_num_threads(6);
	if(img_color)
		sprintf(ext_images, "%s", "ppm");
	else
		sprintf(ext_images, "%s", "pgm");

	sprintf(ext_gt, "%s", "pgm");
	sprintf(dot_ext_images, ".%s", ext_images);
	sprintf(dot_ext_gt, ".%s", ext_gt);

	nimages   = iftCountImagesDirectory(directory_path, ext_images);
	img_names = iftCreateAndLoadImageDirectory(nimages, directory_path, ext_images);

	//nimages_gt   = iftCountImagesDirectory(directory_path_gt, ext_gt);
	//img_names_gt = iftCreateAndLoadImageDirectory(nimages_gt, directory_path_gt, ext_gt);

	printf("Directory: %s\n", directory_path);
	printf("Number of Images: %d\n", nimages);
	puts("");

	DIR* dir = opendir(directory_path);
	if (dir)
	{
		printf("exists\n");
	    /* Directory exists. */
	    closedir(dir);
	}else{
	    /* Directory does not exist. */
		printf("does not exist\n");
	}

	iftRandomSeed(IFT_RANDOM_SEED);

	for(int ind = 0; ind < nimages ; ind++) {
		printf("It. = %d, Image %s, class %d\n", ind, img_names[ind].image_name, 0);

		sprintf(filename, "%s%s", directory_path, img_names[ind].image_name);
		strcpy(img_names[ind].image_name, iftSplitStringOld(img_names[ind].image_name, dot_ext_images, 0));
		strcpy(image_name, img_names[ind].image_name);
		sprintf(filename_output, "%s%s%s%s", directory_path,"output/" ,img_names[ind].image_name,dot_ext_gt);
		sprintf(directory_output, "%s%s", directory_path,"output/");

		//sprintf(filename_gt, "%s%s", directory_path_gt, img_names_gt[ind].image_name);
		//strcpy(img_names_gt[ind].image_name, iftSplitStringOld(img_names_gt[ind].image_name, dot_ext_gt, 0));


		printf("image %s\n",filename);
		//printf("image GT %s\n",filename_gt);

		if(img_color){
			img = iftReadImageP6(filename);
		}else{
			img = iftReadImageP5(filename);
		}
		//gt = iftReadImageP5(filename_gt);


		num_image = iftSelectKernelsFromWGRegions(img, gt, volume_threshold, xsize, ysize, nsamples, AAllow, image_name, directory_output, convnet, type_desc);

		iftDestroyConvNetwork(&convnet);
		iftDestroyImage(&img);
		//iftDestroyMImage(&input);
		//iftDestroyMImage(&output);
		//iftDestroyConvNetwork(&convnet);
	}

	// Deallocators
	iftDestroyImageNames(img_names);

	return -1;
}


