#include <stdio.h>
#include <stdlib.h>

#include "ift.h"
#include <tiffio.h>
#include "iftTIFF.h"

int main(int argc, char* argv[])
{
  if ( (argc != 2) && (argc != 3) ) {
    char tmp[200];
    sprintf(tmp,"usage: %s [input_tiff] <output.pgm>\n",argv[0]);
    iftError(tmp,"opentiff");
  }

  char fileIn[200];
  strcpy(fileIn,argv[1]);
  iftImage* imgIn = TIFFRGB2IFT(fileIn);

  char fileOut[200];
  if (argc == 3)
    strcpy(fileOut,argv[2]);
  else {
    strcpy(fileOut,argv[1]);
    char* p = strrchr(fileOut,'.') + 1;
    strcpy(p,"pgm");
  }

  if (imgIn) {
    if (iftIsColorImage(imgIn)) {
      // iftWriteImageP6(imgIn,"output.ppm");
      iftImage* imgTmp=iftImageGray(imgIn);
      iftDestroyImage(&imgIn);
      imgIn=imgTmp;
    }
    iftWriteImageP5(imgIn,fileOut);
  }
  iftDestroyImage(&imgIn);

  return 0;
}
