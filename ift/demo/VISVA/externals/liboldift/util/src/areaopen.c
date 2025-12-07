
#include "ctree.h"

int main(int argc, char **argv)
{
  Image *img,*fimg;
  int    Area;

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
    fprintf(stderr,"usage: areaopen <image.pgm> <Area>\n");
    exit(-1);
  }

  img  = ReadImage(argv[1]);
  Area = atoi(argv[2]);
  fimg = CTAreaOpen(img,Area);
  WriteImage(fimg,"areaopen.pgm");

  DestroyImage(&img);
  DestroyImage(&fimg);

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
