#include "ift.h"


int main(int argc, char **argv)
{
  CImage *img,*roi;
  

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/
  
  if (argc != 7) {
    fprintf(stderr,"usage: roi <input.ppm> <output.ppm> <xl> <yl> <xr> <yr>\n");
    fprintf(stderr,"(xl,yl): coordinates of the upper-left point\n");
    fprintf(stderr,"(xr,yr): coordinates of the lower-right point\n");
    exit(-1);
  }

  img  = ReadCImage(argv[1]);
  roi  = CROI(img,atoi(argv[3]),atoi(argv[4]),atoi(argv[5]),atoi(argv[6]));
  WriteCImage(roi,argv[2]);

  DestroyCImage(&img);
  DestroyCImage(&roi);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}
