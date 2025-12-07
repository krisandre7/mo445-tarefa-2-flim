#include "ift.h"

int main(int argc, char **argv)
{
  Image  *img[10];
  AdjRel *A=NULL;
  float radius;

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
    fprintf(stderr,"usage: morph <orig.pgm> <radius> \n");
    exit(-1);
  }

  img[0]= ReadImage(argv[1]);
  img[1]= ReadImage(argv[2]);
  img[2]=Leveling(img[0],img[1]);
  WriteImage(img[2],"leveling.pgm");
  exit(0);

  img[0]= ReadImage(argv[1]);
  radius= atof(argv[2]);
  A = Circular(radius);


  img[1] = Threshold(img[0],91,255);
  img[2]= FastAreaOpen(img[1],2000);
  WriteImage(img[2],"mama.pgm");
  img[3]= CloseHoles(img[2]);
  WriteImage(img[3],"closeholes.pgm");
  img[4]= Diff(img[3],img[2]);
  WriteImage(img[4],"residuo.pgm");
  img[5]= OpenRec(img[4],A);
  WriteImage(img[5],"tumor.pgm");

  DestroyImage(&img[0]);
  DestroyImage(&img[1]);
  DestroyImage(&img[2]);
  DestroyImage(&img[3]);
  DestroyImage(&img[4]);
  DestroyImage(&img[5]);
  DestroyAdjRel(&A);
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
