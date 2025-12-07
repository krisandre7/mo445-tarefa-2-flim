
#include "ift.h"

int main(int argc, char **argv)
{
  Image *oimg,*bin,*simg,*binaux;
  Curve3D *salience;

  /*------- -------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/
  
  if (argc != 4) {
    fprintf(stderr,"usage: saliences <orig.pgm> <dest.pgm> <thres in [0,180]>\n");
    exit(-1);
  }

  oimg     = ReadImage(argv[1]);
  binaux   = Threshold(oimg,250,255); // image must be 0 or 1
  DestroyImage(&oimg);
  bin      = AddFrame(binaux,10,0);
  // salience = iftContourSaliences(bin,5,20,70,110); // based on skeletons
  salience = ContourSaliencePoints(bin,atof(argv[3])); // simpler approach
  simg     = PaintSaliences(bin,salience);
  WriteImage(simg,argv[2]);

  DestroyImage(&simg);
  DestroyImage(&bin);
  DestroyCurve3D(&salience);
  DestroyImage(&binaux);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}
