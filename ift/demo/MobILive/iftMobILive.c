#include <stdio.h>
#include <stdlib.h>

#include "ift.h"
#include "tiffio.h"

#define CONVNETNAME "MobILive.convnet"

/************* BODIES *************/
int iftCompareImageName(  void * a,   void * b) {
	iftImageNames *first = (iftImageNames*) a;
	iftImageNames *second = (iftImageNames*) b;
	if ( strcmp(first->image_name, second->image_name) < 0 ) return -1;
	if ( strcmp(first->image_name, second->image_name) == 0 ) return 0;

	return 1;
}

iftImage* TIFFRGB2IFT(char* filename)
{
  TIFF* tif = TIFFOpen(filename, "r");
  if (tif) {
    uint32 w, h;
    size_t npixels;
    uint32* raster;

    iftImage* img = NULL;
    
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
    npixels = w * h;
    raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32));
    if (raster != NULL) {
      if (TIFFReadRGBAImage(tif, w, h, raster, 0)) {
	iftColor RGB,YCbCr;
	img = iftCreateColorImage(w,h,1);
	for(int s=0;s<img->n;s++) {
	  RGB.val[0] = raster[s] >>  0 & 0xFF;
	  RGB.val[1] = raster[s] >>  8 & 0xFF;
	  RGB.val[2] = raster[s] >> 16 & 0xFF;
	  YCbCr      = iftRGBtoYCbCr(RGB);
	  img->val[npixels-s-1] =        YCbCr.val[0];
	  img->Cb [npixels-s-1] =(ushort)YCbCr.val[1];
	  img->Cr [npixels-s-1] =(ushort)YCbCr.val[2];
	}
      }
      _TIFFfree(raster);
    }
    TIFFClose(tif);

    return img;
  } else {
    return NULL;
  }

  return 0;
}

int main(int argc, char* argv[])
{
  if (argc < 2) {
    char tmp[200];
    sprintf(tmp,"usage: %s test_images_path\n",argv[0]);
    iftError(tmp,"opentiff");
  }

  // is argv[1] a directory?
  char directory[200],fullpath[200];
  strcpy(directory,argv[1]);
  if (!iftDirExists(directory))
    iftError("Error: Images directory doesn't exist!!!", "iftMobILive");

  int nimages;
  iftImageNames *img_names;
  nimages = iftCountImageNames(directory,".tiff");
  img_names = iftCreateAndLoadImageNames(nimages, directory,".tiff");
  qsort(img_names, nimages, sizeof(iftImageNames), iftCompareImageName);

  //  load convnet 
  iftConvNetwork *convnet = iftReadConvNetwork(CONVNETNAME);

  //  load svm configuration

  //  open classification.csv file
  FILE* pFcsv = fopen("classification.csv","wt+");
  if ( pFcsv == NULL) {
    iftError("Impossible to create the 'classification.csv' file!","main");
  }

  if (nimages <= 0) {
    iftError("No images found to be processed","iftMobILive");
  }

  fprintf(stdout,"nimages to be processed: %d\n",nimages);
  for (int i = 0; i <nimages ; i++) {
    // Reading TIFF Image and converting it to PGM
    sprintf(fullpath,"%s/%s",directory,img_names[i].image_name);
    iftImage* imgIn = TIFFRGB2IFT(fullpath);
    if (imgIn) {
      if (iftIsColorImage(imgIn)) {
	// iftWriteImageP6(imgIn,"output.ppm");
	iftImage* imgTmp=iftImageGray(imgIn);
	iftDestroyImage(&imgIn);
	imgIn=imgTmp;
      }
      // iftWriteImageP5(imgIn,img_names[i].image_name);
    }
    else {
      fprintf(stdout,"error while reading %s image - deciding for Fake\n",fullpath);
      fprintf(pFcsv ,"%s\tFake\n",img_names[i].image_name);
      continue;
    }

    //    extract feature

    //    apply classification
    int label = 1;

    //    define label
    if (label == 1) {
      fprintf(pFcsv ,"%s\tFake\n",img_names[i].image_name);
      fprintf(stdout,"%s\tFake\n",img_names[i].image_name);
    } else { // (label == 2)
      fprintf(pFcsv ,"%s\tReal\n",img_names[i].image_name);
      fprintf(stdout,"%s\tReal\n",img_names[i].image_name);
    }

    iftDestroyImage(&imgIn);
  }

  fclose(pFcsv);

  iftDestroyImageNames(img_names);

  // release svm classifier

  // release convnet
  iftDestroyConvNetwork(&convnet);

  return 0;
}
