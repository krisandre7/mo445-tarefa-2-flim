#include "ift.h"

Image *ColorEnhanceParasites(CImage *cimg)
{
  Image *img=CreateImage(cimg->C[0]->ncols,cimg->C[0]->nrows);
  int p,n=cimg->C[0]->ncols*cimg->C[0]->nrows;
  CImage *cimg1=CImageRGBtoYCbCr(cimg);

  WriteImage(cimg1->C[1],"Cb.pgm");
  WriteImage(cimg1->C[2],"Cr.pgm");
  for (p=0; p < n; p++) {
    img->val[p]=(int)((float)cimg1->C[2]->val[p]+(float)cimg1->C[1]->val[p])/2;
  }
  
  DestroyCImage(&cimg1);
  return(img);
}

int main(int argc, char **argv)
{
  char filename[200];
  Image *img,*bin,*view;
  AdjRel *A;
  CImage *cimg;

  /*--------------------------------------------------------*/
#ifndef _WIN32
  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;
#endif

  /*--------------------------------------------------------*/

  if (argc != 3) {
    fprintf(stderr,"usage: %s <basename> <adj. radius> \n",argv[0]);
    exit(-1);
  }
  sprintf(filename,"%s.ppm",argv[1]);
  cimg    = ReadCImage(filename);
  view    = ift_CopyImage(cimg->C[0]);
  img     = ColorEnhanceParasites(cimg);
  DestroyCImage(&cimg);
  A       = Circular(atof(argv[2]));
  bin     = iftThres(img,A);
  DestroyAdjRel(&A);
  cimg    = DrawCBorder(view,bin);
  sprintf(filename,"%s_segm.ppm",argv[1]);
  WriteCImage(cimg,filename);
  DestroyImage(&img);
  DestroyImage(&view);
  DestroyImage(&bin);
  DestroyCImage(&cimg);

  /* ---------------------------------------------------------- */

#ifndef _WIN32
  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   
#endif

  return(0);
}

