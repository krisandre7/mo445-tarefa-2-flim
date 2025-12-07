#include "ift.h"
#include "common.h"
//#include "iftImageDirectory.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

int iftSelectBorderPatches(iftImage *img,iftImage *gt, float volume_threshold, int xsize, int ysize, int nsamples, iftAdjRel *AAllow, char *image_name, char *directory_output, int num_image, int num_scales, int convnet_size){
	int* selectedPIX;
	int p,q,r,j,s,center,i,count,k,t;
	char filename_output[400];
	iftImage *basins,*marker,*label_image,*border_wg;
	int scales[num_scales];
	iftAdjRel *APatch,*AAux;
	iftAdjRel *A = iftCircular(sqrtf(2.0));
	iftAdjRel *ACheckWG = iftCircular(sqrtf(2.0));
	p = 0;
	q = 0;
	scales[0] = 7;
	scales[1] = 15;
	scales[2] = 25;
	if(num_scales > 1){
		xsize = scales[num_scales-1];
		ysize = scales[num_scales-1];
	}
	image_name[5] = '1'; //set class 1 XXXXXX_YYYYYYYY.pgm format X class Y num_img

	// get watergray
	basins = iftImageBasins(img,A);
	marker = iftVolumeClose(basins,volume_threshold);
	label_image = iftWaterGray(basins,marker,A);
	// iftWaterGrayBorder removed I dont know by who
	//border_wg = iftWaterGrayBorder(label_image,A);
	border_wg = iftCopyImage(label_image);
	iftDestroyImage(&basins);
	iftDestroyImage(&marker);
	iftDestroyImage(&label_image);


	// get number pixels gt
	int num_gt = 0;
	for(p=0;p<gt->n;p++){
		if(gt->val[p] == 255)
			num_gt++;
	}
	// get gt border pixels
	int *pos_gt = iftAllocIntArray(num_gt);
	j = 0;
	for(p=0;p<gt->n;p++){
		if(gt->val[p] == 255){
			pos_gt[j] = p;
			j++;
		}
	}
	// Select Patches
	selectedPIX = iftAllocIntArray(num_gt);
	APatch = iftRectangular(xsize,ysize);
	j = 0;
	s = 0;

	iftImage *notallowed = iftCreateImage(gt->xsize,gt->ysize,gt->zsize);
	count = 0;
	while(s < nsamples && count < img->n){
		j = iftRandomInteger(0,num_gt-1);
		p = pos_gt[j];
		if(selectedPIX[j] == 0){
			selectedPIX[j] = 1;
			iftVoxel pixel   = iftGetVoxelCoord(img ,p);
			for(i=1;i<APatch->n;i++){
				iftVoxel neighbor   = iftGetAdjacentVoxel(APatch,pixel,i);
				if (!iftValidVoxel(gt,neighbor)){
					break; // Not valid pixel
				}
			}

			for(k=1;k<AAllow->n;k++){  // If it is far enough from another already selected
				iftVoxel neighbor  = iftGetAdjacentVoxel(AAllow,pixel,k);
				if (iftValidVoxel(gt,neighbor)){
					q = iftGetVoxelIndex(gt,neighbor);
					if(notallowed->val[q] == 255){
						break;
					}
				}else{
					break;
				}
			}

			int num_intersec_wg = 0;
			for(t=0;t<ACheckWG->n;t++){  // If it is near to watergray border pixels
				iftVoxel neighbor  = iftGetAdjacentVoxel(ACheckWG,pixel,t);
				if (iftValidVoxel(gt,neighbor)){
					q = iftGetVoxelIndex(gt,neighbor);
					if(border_wg->val[q] == 255){
						num_intersec_wg++;
					}
				}
			}


			if(i==APatch->n && k==AAllow->n && (num_intersec_wg>0)){ // valid Patch
				filename_output[0] = 0;
				notallowed->val[p] = 255;  // mark as notallowed
				for(int is=0;is<num_scales;is++){ // for every scale
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
						q = iftGetVoxelIndex(gt,u);
						iftVoxel v   = iftGetAdjacentVoxel(AAux,vox_center,i);
						r = iftGetVoxelIndex(patch,v);
						patch->val[r] = img->val[q];
						if (img->Cb != NULL) {
							patch->Cb[r]=img->Cb[q];
							patch->Cr[r]=img->Cr[q];
						}
					}
					if (img->Cb != NULL) {
						iftImage* out = iftInterp2D(patch, (float)convnet_size/(float)patch->xsize, (float)convnet_size/(float)patch->ysize);
						sprintf(filename_output, "%s%d/%s%08d%s", directory_output, (is+1) ,"01_",(s+num_image),".ppm");
						iftWriteImageP6(out, filename_output);
						iftDestroyImage(&out);
					}else{
						iftImage* out = iftInterp2D(patch, (float)convnet_size/(float)patch->xsize, (float)convnet_size/(float)patch->ysize);
						sprintf(filename_output, "%s%d/%s%08d%s", directory_output, (is+1) ,"01_",(s+num_image),".pgm");
						iftWriteImageP5(out, filename_output);
						iftDestroyImage(&out);
					}
					iftDestroyImage(&patch);
				}
				s++;
			}
		}
		count ++;
	}
	printf("Selected %d Border Patches\n",s);
	iftDestroyImage(&notallowed);
	iftDestroyAdjRel(&APatch);
	iftDestroyAdjRel(&AAux);
	free(selectedPIX);
	return (s+num_image);
}


int iftSelectNonBorderPatches(iftImage *img,iftImage *gt, float volume_threshold, int xsize, int ysize, int nsamples, iftAdjRel *AAllow, char *image_name, char *directory_output, int num_image, int num_scales,int convnet_size){
	int* selectedPIX;
	int p,q,r,s,center,i,count;
	char filename_output[400];
	int scales[num_scales];
	iftImage *basins,*marker,*label_image;
	iftAdjRel *APatch,*AAux;
	iftAdjRel *A = iftCircular(sqrtf(2.0));
	p = 0;
	q = 0;
	scales[0] = 7;
	scales[1] = 15;
	scales[2] = 25;
	if(num_scales > 1){
		xsize = scales[num_scales-1];
		ysize = scales[num_scales-1];
	}
	image_name[5] = '0'; //set class 0  XXXXXX_YYYYYYYY.pgm format X class Y num_img
	basins = iftImageBasins(img,A);
	marker = iftVolumeClose(basins,volume_threshold);
	label_image = iftWaterGray(basins,marker,A);
	iftDestroyImage(&basins);
	iftDestroyImage(&marker);

	selectedPIX = iftAllocIntArray(img->n);
	APatch = iftRectangular(xsize,ysize);
	s = 0;
	iftImage *notallowed = iftCopyImage(gt);
	count = 0;
	while(s < nsamples && count < img->n ){
		p = iftRandomInteger(0,img->n-1);
		if(selectedPIX[p] == 0){
			selectedPIX[p] = 1;
			iftVoxel pixel   = iftGetVoxelCoord(img ,p);
			for(i=1;i<APatch->n;i++){
				iftVoxel neighbor   = iftGetAdjacentVoxel(APatch,pixel,i);
				if (iftValidVoxel(gt,neighbor)){
					q = iftGetVoxelIndex(gt,neighbor);
					if(label_image->val[p] != label_image->val[q] || notallowed->val[q] == 255 ) { // If some adjacent is of another region is an invalid patch
						break;
					}
				}else{
					break;
				}
			}
			if(i==APatch->n){ // valid Patch
				filename_output[0] = 0;
				notallowed->val[p] = 255;  // mark as notallowed
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
						q = iftGetVoxelIndex(gt,u);
						iftVoxel v   = iftGetAdjacentVoxel(AAux,vox_center,i);
						r = iftGetVoxelIndex(patch,v);
						patch->val[r] = img->val[q];
						if (img->Cb != NULL) {
							patch->Cb[r]=img->Cb[q];
							patch->Cr[r]=img->Cr[q];
						}
					}
					if (img->Cb != NULL) {
						iftImage* out = iftInterp2D(patch, (float)convnet_size/(float)patch->xsize, (float)convnet_size/(float)patch->ysize);
						sprintf(filename_output, "%s%d/%s%08d%s", directory_output, (is+1) , "00_",(s+num_image),".ppm");
						iftWriteImageP6(out, filename_output);
						iftDestroyImage(&out);
					}else{
						iftImage* out = iftInterp2D(patch, (float)convnet_size/(float)patch->xsize, (float)convnet_size/(float)patch->ysize);
						sprintf(filename_output, "%s%d/%s%08d%s", directory_output, (is+1), "00_",(s+num_image),".pgm");
						iftWriteImageP5(out, filename_output);
						iftDestroyImage(&out);
					}
					iftDestroyImage(&patch);
				}
				s++;
			}
		}
		count ++;
	}
	printf("Selected %d Non Border Patches\n",s);
	iftDestroyImage(&label_image);
	iftDestroyImage(&notallowed);
	iftDestroyAdjRel(&APatch);
	iftDestroyAdjRel(&AAux);
	free(selectedPIX);
	return (s+num_image);
}


void iftMultiWaterGray(iftImage *img, float volume_threshold, char *filename_output){
	float volume;
	int p,j,q;
	int iter = 0;
	iftImage *basins,*marker,*label_image;
	iftVoxel u,v;
	iftAdjRel *A = iftCircular(sqrtf(2.0));
	iftImage *grad = iftCreateImage(img->xsize,img->ysize,img->zsize);
	volume = volume_threshold;
	while( iter < 255 ){
		//printf("Iter: %d vol: %f \n",iter,volume);
		volume = volume * (1+(1.0/(float)(iter+1)));
		basins = iftImageBasins(img,A);
		marker = iftVolumeClose(basins,volume);
		label_image = iftWaterGray(basins,marker,A);

		for (p=0; p < img->n; p++) {
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
			  grad->val[p] = grad->val[p] + 1;
			  break;
			}
			  }
			}
		}
		iter++;
		iftDestroyImage(&basins);
		iftDestroyImage(&marker);
		iftDestroyImage(&label_image);
	}
	iftWriteImageP2(grad, filename_output);
	iftDestroyImage(&grad);

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
	if (argc != 9)
		iftError("Please provide the following parameters:\n<IMAGE_DIRECTORY> <IMAGE_DIRECTORY_GT> <COR=1 GRAY=0> <BORDER 1 NB 0> <SIZE> <VOLUME> <#SCALES> <#PATCHES>\n\n", "main");

	//char *file_config = argv[1];
	char *directory_path = argv[1];
	char *directory_path_gt = argv[2];
	int img_color = atoi(argv[3]);
	int type_patch = atoi(argv[4]);
	int size = atoi(argv[5]);
	float volume_threshold = atof(argv[6]);
	int num_scales = atoi(argv[7]);
	int nsamples = atoi(argv[8]);
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
	iftAdjRel *AAllow = iftRectangular(11,11);
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

	nimages   = iftCountImageNames(directory_path, ext_images);
	img_names = iftCreateAndLoadImageNames(nimages, directory_path, ext_images);

	nimages_gt   = iftCountImageNames(directory_path_gt, ext_gt);
	img_names_gt = iftCreateAndLoadImageNames(nimages_gt, directory_path_gt, ext_gt);

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

	for(int ind = 0; ind < nimages ; ind++) {
		printf("It. = %d, Image %s, class %d\n", ind, img_names[ind].image_name, 0);

		sprintf(filename, "%s%s", directory_path, img_names[ind].image_name);
		strcpy(img_names[ind].image_name, iftSplitStringOld(img_names[ind].image_name, dot_ext_images, 0));
		strcpy(image_name, img_names[ind].image_name);
		sprintf(filename_output, "%s%s%s%s", directory_path,"output/" ,img_names[ind].image_name,dot_ext_gt);
		sprintf(directory_output, "%s%s", directory_path,"output/");

		sprintf(filename_gt, "%s%s", directory_path_gt, img_names_gt[ind].image_name);
		strcpy(img_names_gt[ind].image_name, iftSplitStringOld(img_names_gt[ind].image_name, dot_ext_gt, 0));


		printf("image %s\n",filename);
		printf("image GT %s\n",filename_gt);

		if(img_color){
			img = iftReadImageP6(filename);
		}else{
			img = iftReadImageP5(filename);
		}
		gt = iftReadImageP5(filename_gt);

		if(type_patch)
			num_image = iftSelectBorderPatches(img, gt, volume_threshold, xsize, ysize, nsamples, AAllow, image_name , directory_output, num_image, num_scales, convnet_size);
		else
			num_image = iftSelectNonBorderPatches(img, gt, volume_threshold, xsize, ysize, nsamples, AAllow, image_name, directory_output, num_image, num_scales, convnet_size);

		//iftMultiWaterGray(img, volume_threshold, filename_output);
		iftDestroyImage(&img);
		//iftDestroyMImage(&input);
		//iftDestroyMImage(&output);
		//iftDestroyConvNetwork(&convnet);
	}

	// Deallocators
	iftDestroyImageNames(img_names);

	return -1;
}


