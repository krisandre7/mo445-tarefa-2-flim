#include "radiometric3.h"

int main(int argc, char **argv)
{
  Scene *img1,*img2,*mimg;
  Scene *mask1,*mask2;

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
  
  if (argc != 5) {
    fprintf(stderr,"usage: matchhistomasks3 <orig.scn> <orig-mask.scn> <dest.scn> <dest-mask.scn>\n");
    exit(-1);
  }

  img1  = ReadScene(argv[1]);
  mask1 = ReadScene(argv[2]);
  img2  = ReadScene(argv[3]);
  mask2 = ReadScene(argv[4]);
  mimg  = MatchHistogramMasks3(img1,mask1,img2,mask2);
  WriteScene(mimg,"matched.scn");

  DestroyScene(&img1);
  DestroyScene(&img2);
  DestroyScene(&mask1);
  DestroyScene(&mask2);
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
