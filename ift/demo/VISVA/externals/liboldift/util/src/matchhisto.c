
#include "radiometric.h"

int main(int argc, char **argv)
{
  Image *img1,*img2,*mimg;
  Curve *hist1,*hist2;

  /*------- -------------------------------------------------*/
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
    fprintf(stderr,"usage: matchhisto <orig.pgm> <dest.pgm>\n");
    exit(-1);
  }

  img1  = ReadImage(argv[1]);
  img2  = ReadImage(argv[2]);
  mimg  = MatchHistogram(img1,img2);
  hist1 = NormHistogram(mimg);
  hist2 = NormHistogram(img2);
  WriteCurve(hist1,"hist1.txt");
  WriteCurve(hist2,"hist2.txt");
  WriteImage(mimg,"matched.pgm");

  DestroyCurve(&hist1);
  DestroyCurve(&hist2);
  DestroyImage(&img1);
  DestroyImage(&img2);
  DestroyImage(&mimg);

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
