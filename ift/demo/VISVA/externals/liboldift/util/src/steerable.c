#include "ift.h"

int main(int argc, char **argv)
{
  CImage *cimg;
  Features *lab,*stb;
  char filename[100];
  int i;
  DImage *band;
  Image *img;

  /*------- -------------------------------------------------*/

  void *trash = malloc(1);
  struct mallinfo info;
  int MemDinInicial, MemDinFinal;
  free(trash);
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc != 2) {
    fprintf(stderr,"usage: steerable <image.ppm> \n");
    exit(-1);
  }

  cimg = ReadCImage(argv[1]);

  lab  = LabCImageFeats(cimg);
//  NormalizeLab(lab);
  stb = ShiftedSPFeats(lab,3,6);
  for (i=0; i < 54; i++) {
    sprintf(filename,"steerable%02d.pgm",i);
    band = GetFeature(stb,i);
    img  = ConvertDImage2Image(band);
    WriteImage(img,filename);
    DestroyDImage(&band);
    DestroyImage(&img);
  }

  DestroyCImage(&cimg);
  DestroyFeatures(&lab);
  DestroyFeatures(&stb);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);

  return(0);
}
