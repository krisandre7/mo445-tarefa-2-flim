#include "radiometric3.h"

int main(int argc, char **argv)
{
  Scene *img,*eimg,*mask;
  int    Imax;

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
  
  if ( (argc != 3) && (argc != 4) ) {
    fprintf(stderr,"usage: equalize3 <input.scn> <Imax> [<mask.scn>]\n");
    exit(-1);
  }

  img  = ReadScene(argv[1]);
  Imax = atoi(argv[2]);
  if (argc == 4) {
    mask = ReadScene(argv[3]);
    eimg = EqualizeMask3(img,mask,Imax);
  }
  else {
    eimg = Equalize3(img,Imax);
  }
  WriteScene(eimg,"equalized.scn");

  DestroyScene(&mask);
  DestroyScene(&img);
  DestroyScene(&eimg);

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
