#include "ift.h"
#include "common.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

iftImage* iftMergeWatergray(iftImage* image, float volume_threshold, int nregions, char* outputwg){

	iftAdjRel *adj1 = iftCircular(sqrtf(2.0));
	iftAdjRel *adj2 = iftCircular(1.0);

	iftImage *basins = iftImageBasins(image,adj1);
	iftImage *marker = iftVolumeClose(basins, volume_threshold*2);

	iftImage *wg_label_image = iftWaterGray(basins,marker,adj1);

	//iftDataSet* dataset = iftSupervoxelsToMeanStdSizeDataset(image, wg_label_image);
	iftDataSet* dataset = iftSupervoxelsToSelectiveSearchDataset(image, wg_label_image, 25);

	float alpha[10];
	alpha[0] = 0.2;
	alpha[1] = 1.0;
	alpha[2] = 1.0;
	alpha[3] = 0.2;
	alpha[4] = 1.0;
	alpha[5] = 1.0;
	alpha[6] = 1.0;
	alpha[7] = 0.0;
	alpha[8] = 0.0;
	alpha[9] = 0.9;

	//iftRegionHierarchy *rh = iftCreateRegionHierarchy(wg_label_image, dataset, adj2, iftDistMeanStdSizeSupervoxel, iftMergeMeanStdSizeSupervoxel, alpha);
	iftRegionHierarchy *rh = iftCreateRegionHierarchy(wg_label_image, dataset, adj2, iftDistSelectiveSearchSupervoxel, iftMergeSelectiveSearchSupervoxel, alpha);
	iftImage *label_image = iftFlattenRegionHierarchy(rh, nregions);


	iftImage *label = iftCopyImage(label_image);
	iftImage *clon = iftCopyImage(image);
	iftAdjRel *adj3 = iftCircular(0.0);
	iftColor RGB, YCbCr;
	RGB.val[0] = 0;
	RGB.val[1] = 0;
	RGB.val[2] = 255;
	YCbCr      = iftRGBtoYCbCr(RGB);
	iftDrawBorders(clon,label,adj1,YCbCr,adj3);
	iftWriteImageP6(clon, outputwg);
	iftDestroyImage(&clon);


	iftDestroyImage(&wg_label_image);
	iftDestroyAdjRel(&adj1);
	iftDestroyAdjRel(&adj2);
	iftDestroyAdjRel(&adj3);
	iftDestroyImage(&basins);
	iftDestroyImage(&marker);
	iftDestroyDataSet(&dataset);
	iftDestroyRegionHierarchy(&rh);

	return label_image;
}

iftImage *iftDrawRegionBorders(iftImage *img, iftImage *label, iftAdjRel *A)
{
  iftVoxel u,v;
  int i,p,q;

  iftImage *points = iftCreateImage(img->xsize,img->ysize,img->zsize);
  for (p=0; p < img->n; p++) {
    u.x = iftGetXCoord(label,p);
    u.y = iftGetYCoord(label,p);
    u.z = iftGetZCoord(label,p);
    for (i=0; i < A->n; i++) {
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (iftValidVoxel(label,v)){
	q = iftGetVoxelIndex(label,v);
	if (label->val[p] < label->val[q]){
	  points->val[p] = 255;
	  break;
	}
      }
    }
  }
  return points;
}

float iftVerifyBorders(iftImage *imgBorder, iftImage *label, iftAdjRel *A, char *filename_points)
{
  iftVoxel u,v;
  int i,p,q,numBorders,numLabelBorders;
  numBorders = 0;
  numLabelBorders = 0;
  iftImage *regionBorders = iftDrawRegionBorders(imgBorder, label, A);
  iftImage *points = iftCreateImage(imgBorder->xsize,imgBorder->ysize,imgBorder->zsize);
  for (p=0; p < imgBorder->n; p++) {
    u.x = iftGetXCoord(label,p);
    u.y = iftGetYCoord(label,p);
    u.z = iftGetZCoord(label,p);
    if(imgBorder->val[p] == 255){
    	numBorders++;
    }
    for (i=0; i < A->n; i++) {
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (iftValidVoxel(label,v)){
	q = iftGetVoxelIndex(label,v);
	if ((imgBorder->val[p] == 255) && (regionBorders->val[q] == 255)){
		numLabelBorders++;
		points->val[p] = 255;
	  break;
	}
      }
    }
  }
  iftWriteImageP5(points,filename_points);
  iftDestroyImage(&points);
  printf("B/L:  %d/%d \n",numBorders,numLabelBorders);
  return (float)numLabelBorders/(float)numBorders;
}

void iftCountSupervoxels(iftImage* gt, iftImage* label_image){
	int *numType;
	int p,numObj,numBg,numInter;
	int nregions = iftMaximumValue(label_image);
	numObj = 0;
	numBg = 0;
	numInter = 0;
	numType = iftAllocIntArray(nregions);
	// initialize
	for (p=0; p < nregions; p++) {
		numType[p] = -1;
	}
	// fill with labels
	for (p=0; p < gt->n; p++) {
		if(numType[label_image->val[p]-1] >= 0){
			if(gt->val[p] != numType[label_image->val[p]-1] && numType[label_image->val[p]-1] != 2){
				numType[label_image->val[p]-1] = 2;
				numInter++;
			}
		}else{
			numType[label_image->val[p]-1] = gt->val[p];
			if(gt->val[p] == 0)
				numBg++;
			else
				numObj++;
		}
	}
	printf("Obj: %d Bg: %d Inter: %d \n",numObj,numBg,numInter);
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
	if (argc != 3)
		iftError("Please provide the following parameters:\n<IMAGE_DIRECTORY> <VOLUME>\n\n", "main");


	char *directory_path = argv[1];
	float volume_threshold = atof(argv[2]);
	int nimages,ntests,nsteps;
	iftImageNames *img_names;
	iftImage *img,*border,*gt;
	char filename[200];
	char filename_border[200];
	char filename_gt[200];
	char filename_output[200];
	char filename_points[200];
	char directory_output[200];

	float ratio = 2;//sqrtf(2.0);
	iftAdjRel *A = iftCircular(ratio);

	omp_set_num_threads(6);

	nimages   = iftCountImageNames(directory_path, ".ppm");
	img_names = iftCreateAndLoadImageNames(nimages, directory_path, ".ppm");
	ntests = 15;
	nsteps = 40;

	printf("Directory: %s\n", directory_path);
	printf("Number of Images: %d\n", nimages);
	puts("");

	//iftRandomSeed(IFT_RANDOM_SEED);

	for(int ind = 0; ind < nimages ; ind++) {
		printf("It. = %d, Image %s, class %d\n", ind, img_names[ind].image_name, 0);

		sprintf(filename, "%s%s", directory_path, img_names[ind].image_name);
		strcpy(img_names[ind].image_name, iftSplitStringOld(img_names[ind].image_name, ".ppm", 0));
		sprintf(directory_output, "%s%s", directory_path,"output/");
		sprintf(filename_border, "%s%s%s%s", directory_path, "border/" ,img_names[ind].image_name,".pgm");
		sprintf(filename_gt, "%s%s%s%s", directory_path, "gt/" ,img_names[ind].image_name,".pgm");

		printf("image %s\n",filename);
		img = iftReadImageP6(filename);
		border = iftReadImageP5(filename_border);
		gt = iftReadImageP5(filename_gt);

		int nregions = 700;
		for(int i = 0; i < ntests ; i++) {
			sprintf(filename_output, "%s%s%s%d%s%d%s", "outputmerge/" ,img_names[ind].image_name,"_",i,"_",nregions,".ppm");
			sprintf(filename_points, "%s%s%s%d%s%d%s", "outputpoints/" ,img_names[ind].image_name,"_",i,"_",nregions,".pgm");
			iftImage *label_image = iftMergeWatergray(img, volume_threshold, nregions, filename_output);
			float rate = iftVerifyBorders(border, label_image, A, filename_points);
			printf("%s: %f\n", filename_output, rate);
			iftCountSupervoxels(gt,label_image);
			nregions = nregions - nsteps;
			iftDestroyImage(&label_image);
		}
	}
	// Deallocators
	iftDestroyImageNames(img_names);

	return -1;
}


