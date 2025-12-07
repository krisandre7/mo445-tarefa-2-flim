#include "radiometric3.h"

int main(int argc, char **argv)
{
  Scene *img1,*img2,*mimg;

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
    fprintf(stderr,"usage: matchhisto3 <orig.scn> <dest.scn>\n");
    exit(-1);
  }

  img1  = ReadScene(argv[1]);
  img2  = ReadScene(argv[2]);
  mimg  = MatchHistogram3(img1,img2);
  WriteScene(mimg,"matched.scn");

  DestroyScene(&img1);
  DestroyScene(&img2);
  DestroyScene(&mimg);

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
