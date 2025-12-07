#include "ift.h"

iftImage *iftSegmentForeground(iftImage *orig, float closing_radius)
{
  iftImage *foreground = iftCreateImage(orig->xsize,orig->ysize,orig->zsize);
 
  for (int z=0; z < orig->zsize; z++) {
    iftImage *slice1   = iftGetXYSlice(orig,z);
    iftImage *slice2   = iftAddFrame(slice1,10,0);
    iftImage *cbasins  = iftCloseBasins(slice2,NULL,NULL);
    iftDestroyImage(&slice1);
    iftDestroyImage(&slice2);
    iftImage *bin1     = iftThreshold(cbasins,0.5*iftOtsu(cbasins),IFT_INFINITY_INT,255);
    iftDestroyImage(&cbasins);
    iftImage *bin2     = iftRemFrame(bin1,10);
    iftPutXYSlice(foreground,bin2,z);
    iftDestroyImage(&bin1);
    iftDestroyImage(&bin2);
  }
  iftAdjRel *A   = iftSpheric(sqrtf(3.0));
  iftImage *bin1 = iftSelectLargestComp(foreground,A);
  iftDestroyAdjRel(&A);
  iftDestroyImage(&foreground);
  iftImage *bin2 = iftAddFrame(bin1,iftRound(closing_radius),0);
  iftDestroyImage(&bin1);
  bin1           = iftCloseBin(bin2,closing_radius);
  iftDestroyImage(&bin2);
  foreground     = iftRemFrame(bin1,iftRound(closing_radius));
  iftDestroyImage(&bin1);

  iftCopyVoxelSize(orig,foreground);
  
  return(foreground);
}


int main(int argc, char *argv[]) 
{
  iftImage       *orig, *fg;
  timer          *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=4)
    iftError("Usage: iftSegmentForeground <original.scn (input)> <foreground.scn (output)> <closing_radius (e.g., 15)>","main");

  orig   = iftReadImageByExt(argv[1]);    
  
  t1     = iftTic();

  fg = iftSegmentForeground(orig,atof(argv[3]));

  iftDestroyImage(&orig);

  t2     = iftToc();
  fprintf(stdout,"Foreground segmented in %f ms\n",iftCompTime(t1,t2));

  iftWriteImageByExt(fg,argv[2]);

  iftDestroyImage(&fg);  
 
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}

