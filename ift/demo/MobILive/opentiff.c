#include <stdio.h>
#include <stdlib.h>

#include "ift.h"
#include "tiffio.h"

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
    fprintf(stdout,"Fake\n" );
    return NULL;
  }

  return 0;
}

int main(int argc, char* argv[])
{
  if (argc < 2) {
    char tmp[200];
    sprintf(tmp,"usage: %s input_image.tiff\n",argv[0]);
    iftError(tmp,"opentiff");
  }

  iftImage* imgIn = TIFFRGB2IFT(argv[1]);
  if (imgIn) {
    if (iftIsColorImage(imgIn)) {
      iftWriteImageP6(imgIn,"output.ppm");
      iftImage* imgTmp=iftImageGray(imgIn);
      iftDestroyImage(&imgIn);
      imgIn=imgTmp;
    }
    iftWriteImageP5(imgIn,"output.pgm");

    iftDestroyImage(&imgIn);
  }

  return 0;
}
