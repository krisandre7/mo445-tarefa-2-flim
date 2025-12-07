#include "common.h"
#include "ift.h"
#include "iftTrainPixels.h"

iftImage* iftGTImageFromRegions(iftImage *label, iftAdjRel *A)
{
  iftVoxel u,v;
  int i,p,q;

  iftImage *img = iftCreateImage(label->xsize,label->ysize,label->zsize);

  if ((img->xsize != label->xsize)||
      (img->ysize != label->ysize)||
      (img->zsize != label->zsize))
    iftError("Images must have the same domain","iftCheckBorders");

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
	if ((label->val[p] < label->val[q]) && (label->val[p] == 255 ||  label->val[p] == 0) ){
		img->val[p] = 255;
	  //br++;
	  break;
	}else{
		img->val[p] = 0;
	}
      }
    }
    /*
    if(img->val[p] == 255){
	  numBorders++;
	  if(br>0){
		  count++;
	  }
	}
	*/
  }
  //printf("match: %d, Total: %d \n",count,numBorders);
  //return ((float)count/(float)numBorders);
  return img;
}

int main(int argc, char **argv) 
{
	if (argc != 2)	iftError("Please provide the following parameters:\n <FILE_INPUT_NAMES>\n\n", "main");

	char *file_inputImages;
	file_inputImages   = argv[1];

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

	iftAdjRel *A = iftCircular(sqrtf(2));
	int idxImg;
	// For all input images
	for( idxImg = 0; idxImg < nimages; idxImg++)
	{
		// Read input images
		fullPath[0] = 0;
		strcpy(fullPath, folder_name);
		strcat(fullPath, files[idxImg].filename);
		printf("Reading : %s\n", files[idxImg].filename);
		iftImage *img = iftReadImageP5(fullPath);
		iftImage *gt = iftGTImageFromRegions(img,A);
	  	strcat(fullPath, ".gt.pgm");
		iftWriteImageP5(gt,fullPath);
	}


	return 0;
}
