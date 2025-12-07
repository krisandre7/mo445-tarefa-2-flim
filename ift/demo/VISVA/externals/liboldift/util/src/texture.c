#include "radiometric.h"
#include "tira.h"

int main(int argc, char **argv)
{
  Image *img1,*img2,*eimg1,*eimg2;
  Vc    *feat1,*feat2;

  /*------- -------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/
  
  if (argc != 5) {
    fprintf(stderr,"usage: texture <img1.pgm> <img2.pgm> <scales> <orientations>\n");
    exit(-1);
  }

  img1  = ReadImage(argv[1]);
  img2  = ReadImage(argv[2]);
  eimg1 = Equalize(img1,255);
  eimg2 = Equalize(img2,255);
  feat1 = Tira(eimg1,atoi(argv[3]),atoi(argv[4]));
  feat2 = Tira(eimg2,atoi(argv[3]),atoi(argv[4]));
  // Distancia
  AlinhaDescritor(feat1);
  AlinhaDescritor(feat2);
  printf("Distance %lf \n",CalcDistEuclid(feat1,feat2));

  DestroyImage(&img1);
  DestroyImage(&eimg1);
  DestroyImage(&img2);
  DestroyImage(&eimg2);
  DestroyVc(&feat1);
  DestroyVc(&feat2);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}
