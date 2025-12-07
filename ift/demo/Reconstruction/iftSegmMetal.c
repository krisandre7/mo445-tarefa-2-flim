#include "ift.h"

int main(int argc, char *argv[]) 
{
  iftImage  *orig, *metal;
  iftAdjRel *A = iftCircular(1.5); 
  timer     *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=2)
    iftError("Usage: iftSegmMetal <projection.scn>","main");
  
  t1          = iftTic();

  orig        = iftReadImage(argv[1]);
  metal       = iftCreateImage(orig->xsize,orig->ysize,orig->zsize);
  for (int z=0; z < orig->zsize; z++) {
    printf("Projection %d\n",z+1);
    iftImage *slice    = iftGetXYSlice(orig,z); 
    iftImage *basins   = iftImageBasins(slice,A);
    iftImage *marker   = iftVolumeClose(basins,10000);
    iftImage *label    = iftWaterGray(basins,marker,A);
    iftPutXYSlice(metal,label,z);
    iftDestroyImage(&slice);
    iftDestroyImage(&basins);
    iftDestroyImage(&marker); 
    iftDestroyImage(&label); 
  }

  t2     = iftToc();
  fprintf(stdout,"Metal segmented in %f ms\n",iftCompTime(t1,t2));

  iftWriteImage(metal,"metal_bin.scn");

  iftDestroyImage(&orig);
  iftDestroyImage(&metal);
  iftDestroyAdjRel(&A);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);

}
