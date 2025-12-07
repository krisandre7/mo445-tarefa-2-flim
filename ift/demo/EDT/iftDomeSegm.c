#include "ift.h"


int main(int argc, char *argv[]) 
{
  iftImage    *bin=NULL,*dist=NULL, *label=NULL;
  iftAdjRel   *A=NULL;

  /*--------------------------------------------------------*/

  int MemDinInicial, MemDinFinal;
  MemDinInicial = iftMemoryUsed();

  /*--------------------------------------------------------*/


  if (argc!=4)
      iftError("Usage: iftDomeSegm <binary.scn> <label.scn> <h (h-dome)>", "main");

  bin   = iftReadImage(argv[1]);
  A     = iftSpheric(sqrtf(3.0));
  dist  = iftEuclDistTrans(bin, A, IFT_INTERIOR, NULL, NULL, NULL);
  label = iftWaterDist(dist,bin,atoi(argv[3]),A);
  iftDestroyImage(&dist);
  iftDestroyImage(&bin);
  iftDestroyAdjRel(&A);
  iftWriteImage(label,argv[2]);
  iftDestroyImage(&label);

  /* ---------------------------------------------------------- */

  MemDinFinal = iftMemoryUsed();
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   


  return(0);
}
