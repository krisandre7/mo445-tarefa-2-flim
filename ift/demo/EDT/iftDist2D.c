#include "ift.h"

int main(int argc, char *argv[]) 
{
  iftImage    *bin=NULL,*dist=NULL;
  iftAdjRel   *A=NULL;
  iftFImage   *aux=NULL;

  /*--------------------------------------------------------*/

  int MemDinInicial, MemDinFinal;
  MemDinInicial = iftMemoryUsed();

  /*--------------------------------------------------------*/


  if (argc!=2)
      iftError("Usage must be: iftDist2D <binary.pgm>", "main");

  bin   = iftReadImageP5(argv[1]);    
  A     = iftCircular(sqrtf(2.0));
  dist  = iftEuclDistTrans(bin, A, IFT_INTERIOR, NULL, NULL, NULL);
  aux   = iftSQRT(dist);  
  iftDestroyImage(&dist);
  dist  = iftFImageToImage(aux,4095);
  iftWriteImageP5(dist,"dist.pgm");

  iftDestroyAdjRel(&A);
  iftDestroyImage(&dist);  
  iftDestroyFImage(&aux);  
  iftDestroyImage(&bin);  
 
  /* ---------------------------------------------------------- */

  MemDinFinal = iftMemoryUsed();
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}



