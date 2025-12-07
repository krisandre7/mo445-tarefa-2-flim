#include "ift.h"


int main(int argc, char *argv[]) 
{
  iftImage       *bin=NULL, *medial_axis=NULL, *rec=NULL;
  timer          *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc!=3)
    iftError("Usage: iftShapeRec2D <binary.pgm> <scale_thres in (0,100] (e.g., 5.0)>","main");

  // MEDIAL_AXIS

  bin    = iftReadImageByExt(argv[1]);
  iftImage *aux = iftThreshold(bin,127,255,255);
  iftDestroyImage(&bin);
  bin  = aux;
    
  t1     = iftTic();
  medial_axis    = iftMedialAxisTrans2D(bin,atof(argv[2]),IFT_INTERIOR);
  t2     = iftToc();
  fprintf(stdout,"MEDIAL_AXIS in %f ms\n",iftCompTime(t1,t2));

  t1     = iftTic();
  rec    = iftShapeReconstruction(medial_axis);
  t2     = iftToc();
  fprintf(stdout,"Shape reconstruction in %f ms\n",iftCompTime(t1,t2));


  iftWriteImageP2(medial_axis,"medial_axis.pgm"); 
  iftWriteImageP2(rec,"result.pgm");
  
  iftDestroyImage(&bin);  
  iftDestroyImage(&medial_axis); 
  iftDestroyImage(&rec);
 
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}



