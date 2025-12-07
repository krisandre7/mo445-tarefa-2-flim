#include "ift.h"


int main(int argc, char *argv[]) 
{
  iftImage    *img=NULL,*dist=NULL;
  iftAdjRel   *A=NULL;
  iftFImage   *aux=NULL;

  /*--------------------------------------------------------*/
  int MemDinInicial, MemDinFinal;
  MemDinInicial = iftMemoryUsed();

  /*--------------------------------------------------------*/


  if (argc!=2)
      iftError("Usage must be: iftDist <image.scn>", "main");

  img   = iftReadImage(argv[1]);    
  A     = iftSpheric(sqrtf(3.0));
  dist  = iftEuclDistTrans(img, A, IFT_INTERIOR, NULL, NULL, NULL);
  aux   = iftSQRT(dist);
  iftDestroyImage(&dist);
  dist  = iftFImageToImage(aux,(int)iftFMaximumValue(aux)+1);
  iftCopyVoxelSize(img,dist);
  iftWriteImage(dist,"dist.scn");

  iftDestroyAdjRel(&A);
  iftDestroyImage(&dist);  
  iftDestroyFImage(&aux);  
  iftDestroyImage(&img);  
 
  /* ---------------------------------------------------------- */

  MemDinFinal = iftMemoryUsed();
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}




