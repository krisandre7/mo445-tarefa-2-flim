#include "ift.h"
#include "common.h"
//#include "iftImageDirectory.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


iftImage* iftWaterGrayBorder(iftImage *label_image,iftAdjRel* A){
  int p,j,q;
  iftVoxel u,v;
  iftImage *grad = iftCreateImage(label_image->xsize,label_image->ysize,label_image->zsize);
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
	  grad->val[p] = 255;
	  break;
	}
      }
    }
  }
  return grad;
}

iftImage* iftMultiWaterGray(iftImage *img, float volume_threshold, iftAdjRel *A){
	float volume,base;
	int p,j,q;
	int iter = 0;
	iftImage *basins,*marker,*label_image;
	iftVoxel u,v;
	iftImage *grad = iftCreateImage(img->xsize,img->ysize,img->zsize);
	volume = volume_threshold;
	base = 4.3;
	while( iter < 10 ){
		base = base - 0.4;
		if(iter > 1){
		volume = volume * (base+(1.0/(float)(iter+1)));
		}
		printf("Iter: %d vol: %9.2f\t",iter,volume);
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
			  grad->val[p] = grad->val[p] + 10;
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
	printf("\n");
	return grad;

}

iftImage* iftDrawWaterGray(iftImage *img, iftImage *label_image){
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
	iftDestroyImage(&label);
	return clon;
}


int main(int argc, char **argv) {
	if (argc != 5)
		iftError("Please provide the following parameters:\n<IMAGE_DIRECTORY> <OUTPUT_PATH> <VOLUME> <NTHREADS>\n\n", "main");

	char *directory_path = argv[1];
	char *output_path = argv[2];
	float volume_threshold = atof(argv[3]);
	int nimages;

	iftImageNames *img_names;

	omp_set_num_threads(atoi(argv[4]));
	//	omp_set_num_threads(6);

	nimages   = iftCountImagesDirectory(directory_path, ".ppm");
	img_names = iftCreateAndLoadImageDirectory(nimages, directory_path, ".ppm");

	printf("Directory: %s\n", directory_path);
	printf("Number of Images: %d\n", nimages);
	puts("");

#pragma omp parallel for shared(img_names,directory_path,output_path,volume_threshold)
	for(int ind = 0; ind < nimages ; ind++) {
	  int p;
	  char image_name[200];
	  char filename[200],filename_output[200];
	  iftImage *basins,*marker,*label_image,*border_wg,*multi_wg;
	  iftImage *img,*new_basins,*orig;

	  iftAdjRel *A = iftCircular(sqrtf(2.0));
	  iftAdjRel *Adjust = iftCircular(30);
	  
	  printf("It. = %d, Image %s, class %d\n", ind, img_names[ind].image_name, 0);

	  sprintf(filename, "%s%s", directory_path, img_names[ind].image_name);
	  strcpy(image_name, iftSplitStringOld(img_names[ind].image_name,".", 0));
	  printf("image_name : %s \n",image_name);
	  sprintf(filename_output, "%s%s%s", output_path, image_name,".pgm");


	  orig = iftReadImageP6(filename);
	  img = iftSmoothImage(orig,Adjust,30);

	  basins = iftImageBasins(img,A);
	  marker = iftVolumeClose(basins,volume_threshold);
	  label_image = iftWaterGray(basins,marker,A);

	  border_wg = iftWaterGrayBorder(label_image, A);
	  multi_wg = iftMultiWaterGray(img, volume_threshold, A);

	  new_basins = iftCreateImage(img->xsize,img->ysize,img->zsize);
	  for(p=0;p<img->n;p++){
	    int maxval = iftMaximumValue(multi_wg);
	    if(multi_wg->val[p] > 0){
	      new_basins->val[p] = (int)((float)basins->val[p] * ((float)multi_wg->val[p]/(float)maxval));
	    }
	  }

	  iftWriteImageP2(new_basins, filename_output);

	  iftDestroyImage(&new_basins);
	  iftDestroyImage(&img);

	  //output = iftImageBasins(new_basins,A);

	  //iftDestroyImage(&marker);
	  //iftDestroyImage(&label_image);

	  //marker = iftVolumeClose(new_basins,volume_threshold);
	  //label_image = iftWaterGray(new_basins,marker,A);

	  //border_draw = iftDrawWaterGray(img,label_image);
	  //output = iftMultiWaterGray(img, volume_threshold, A);

	  //iftDestroyImage(&output);

	  iftDestroyAdjRel(&A);
	  iftDestroyAdjRel(&Adjust);
	}

	// Deallocators
	iftDestroyImageNames(img_names);
	return -1;
}
